import asyncio
import base64
from pathlib import Path

from textual import on
from textual import work
from textual.app import App
from textual.app import ComposeResult
from textual.containers import Horizontal
from textual.containers import Vertical
from textual.containers import VerticalScroll
from textual.validation import Integer
from textual.widgets import Button
from textual.widgets import Footer
from textual.widgets import Header
from textual.widgets import Input
from textual.widgets import Label
from textual.widgets import Select
from textual.widgets import Static
from textual.widgets import Switch
from textual.widgets import TextArea
from textual.worker import get_current_worker
from textual.worker import Worker
from textual_image.widget import Image as TextualImage

from viz.config import Config
from viz.modals import DebugModal
from viz.modals import FileSelectModal
from viz.modals import SessionSelectModal
from viz.services import SessionManager
from viz.services import stream_ollama_chat_async
from viz.tools import ToolRegistry
from viz.widgets import ChatMessage
from viz.widgets import CommandApproval
from viz.workspace import ImageWorkspace


class OllamaVisionApp(App):
    CSS_PATH = "styles.tcss"

    BINDINGS = [
        ("ctrl+enter", "ask_ollama", "Ask / Send"),
        ("ctrl+o", "toggle_view", "Toggle View"),
        ("ctrl+l", "load_session", "Sessions"),
        ("ctrl+2", "toggle_md_view", "Toggle MD"),
        ("ctrl+1", "show_debug", "Debug API"),
        ("escape", "cancel_stream", "Cancel"),
        ("ctrl+q", "quit", "Quit"),
    ]

    def __init__(self, model: str, config: Config) -> None:
        super().__init__()
        self.model = model
        self.config = config
        self.title = f"Ollama Vision ({self.model})"
        self.workspace = ImageWorkspace(self.config)
        self.tool_registry = ToolRegistry(self.config)
        self.session_manager = SessionManager(self.config)
        self.session_id: str = self.session_manager.create_session_id()

        self.messages: list[dict] = []
        self.sent_images: set[Path] = set()
        self.active_worker: Worker | None = None
        self.streaming_msg_widget: ChatMessage | None = None
        self.last_user_msg_widget: ChatMessage | None = None
        self.last_api_payload: dict = {}

        # "prep" or "chat"
        self.view = "prep"
        self.md_enabled: bool = True

    # --- UI Component Properties for DRY access ---
    @property
    def chat_scroll(self) -> VerticalScroll:
        return self.query_one("#chat-scroll", VerticalScroll)

    @property
    def prompt_input(self) -> TextArea:
        return self.query_one("#prompt-input", TextArea)

    @property
    def preset_select(self) -> Select:
        return self.query_one("#preset-select", Select)

    def compose(self) -> ComposeResult:
        yield Header()
        yield from self._compose_prep_panel()
        yield from self._compose_chat_panel()
        yield Footer()

    def _set_view(self, view: str) -> None:
        """Switches visibility between the prep panel and chat panel."""
        self.view = view
        self.query_one("#prep-panel", Vertical).display = (view == "prep")
        self.query_one("#chat-panel", Vertical).display = (view == "chat")

        if view == "chat":
            self.chat_scroll.scroll_end(animate=False)
        elif view == "prep":
            self.prompt_input.focus()

    def _compose_prep_panel(self) -> ComposeResult:
        with Vertical(id="prep-panel"):
            yield TextualImage(id="preview-image")

            with Horizontal(classes="btn-row"):
                yield Button("Capture", id="btn-capture", variant="primary")
                yield Button("Paste", id="btn-paste")
            with Horizontal(classes="btn-row"):
                yield Button("Add File", id="btn-add-file")
                yield Button("Clear", id="btn-clear", variant="warning")

            yield Label("System Message:")
            yield Select(
                self.config.system_presets,
                id="system-select",
                allow_blank=True,
            )

            yield Label("Select Preset:")
            yield Select(
                self.config.presets,
                value="Custom",
                id="preset-select",
                allow_blank=False,
            )

            yield Label("Context Size:")
            yield Input(value=str(self.config.app.default_context),
                        type="integer",
                        id="ctx-input",
                        validators=[Integer(minimum=2048)])

            with Horizontal(classes="toggle-row"):
                yield Label("Enable Tool Calling:")
                yield Switch(value=True, id="tools-toggle")

            yield Label("Prompt:", id="prompt-label")
            yield TextArea(id="prompt-input")

            with Horizontal(classes="btn-row"):
                yield Button(
                    "Ask Ollama",
                    id="btn-ask",
                    variant="success",
                )
                yield Button("Load Chat", id="btn-load", variant="primary")
                yield Button("Reset Chat", id="btn-reset", variant="error")

    def _compose_chat_panel(self) -> ComposeResult:
        with Vertical(id="chat-panel"):
            with VerticalScroll(id="chat-scroll"):
                yield Static(
                    "Attach images and enter a prompt on the left to start chatting.",
                    id="empty-chat-hint",
                )

    def on_mount(self) -> None:
        self.update_preview()
        self._set_view("prep")
        self.prompt_input.focus()

    def _auto_save_session(self) -> None:
        """Automatically saves the current conversation state (text-only) to disk."""
        if not self.messages:
            return

        raw_sys_val = self.query_one("#system-select", Select).value
        system_prompt = "" if raw_sys_val == Select.BLANK else str(raw_sys_val)

        ctx_val = self.query_one("#ctx-input", Input).value
        num_ctx = (
            int(ctx_val)
            if ctx_val.isdigit()
            else self.config.app.default_context
        )

        self.session_manager.save_session(
            session_id=self.session_id,
            model=self.model,
            messages=self.messages,
            system_prompt=system_prompt,
            num_ctx=num_ctx,
        )

    def action_toggle_md_view(self) -> None:
        self.md_enabled = not self.md_enabled
        for msg in self.query(ChatMessage):
            msg.toggle_view(self.md_enabled)

    def update_preview(self) -> None:
        png_files = self.workspace.get_attached_images()
        count = len(png_files)
        img_widget = self.query_one("#preview-image", TextualImage)

        if count == 0:
            img_widget.image = None
            img_widget.display = False
            self.sub_title = "0 images attached"
            return

        img_widget.display = True
        self.sub_title = f"{count} image(s) attached"
        img_widget.image = self.workspace.build_preview()

    def _handle_image_action(self,
                             success: bool,
                             success_msg: str,
                             error_msg: str,
                             severity: str = "error") -> None:
        """DRY helper for updating UI state after an image collection action."""
        if success:
            self.notify(success_msg)
        else:
            self.notify(error_msg, severity=severity)
        self.update_preview()

    @on(Select.Changed, "#preset-select")
    def on_preset_changed(self, event: Select.Changed) -> None:
        if event.value == "Custom":
            self.prompt_input.disabled = False
        else:
            self.prompt_input.text = str(event.value)

    @on(Button.Pressed, "#btn-capture")
    def handle_capture(self) -> None:
        with self.suspend():
            captured = self.workspace.capture_screen_region()
        self._handle_image_action(captured, "Screen region captured.",
                                  "Capture cancelled.", "warning")

    @on(Button.Pressed, "#btn-paste")
    def handle_paste(self) -> None:
        self._handle_image_action(self.workspace.paste_clipboard_image(),
                                  "Image pasted from clipboard.",
                                  "No image found in clipboard.")

    @on(Button.Pressed, "#btn-add-file")
    def handle_add_file(self) -> None:
        self.push_screen(FileSelectModal(), self._on_files_selected)

    def _on_files_selected(self, files: list[Path] | None) -> None:
        if not files:
            return
        added = self.workspace.import_files(files)
        self._handle_image_action(added > 0, f"Added {added} image(s).",
                                  "Failed to convert selected file(s).")

    @on(Button.Pressed, "#btn-clear")
    def handle_clear(self) -> None:
        self.workspace.clear_images()
        self.update_preview()
        self.notify("Cleared attached images.")

    @on(Button.Pressed, "#btn-load")
    def action_load_session(self) -> None:
        """Opens the modal to browse and load saved sessions."""
        self.push_screen(
            SessionSelectModal(self.session_manager),
            self._on_session_selected,
        )

    async def _on_session_selected(self, session_data: dict | None) -> None:
        if not session_data:
            return

        if self.active_worker and self.active_worker.is_running:
            await self.action_cancel_stream()

        self.session_id = session_data.get(
            "id", self.session_manager.create_session_id()
        )
        loaded_messages = session_data.get("messages", [])
        self.messages = []
        self.sent_images.clear()

        # Restore context size if saved
        if "num_ctx" in session_data:
            self.query_one("#ctx-input", Input).value = str(
                session_data["num_ctx"]
            )

        # Restore system preset if it matches an available option
        saved_sys = session_data.get("system_prompt", "")
        sys_select = self.query_one("#system-select", Select)
        valid_sys_values = {val for _, val in self.config.system_presets}
        if saved_sys in valid_sys_values:
            sys_select.value = saved_sys
        elif not saved_sys:
            sys_select.value = Select.BLANK

        await self.chat_scroll.remove_children()

        css_map = {
            "user": "user-msg",
            "assistant": "assistant-msg",
            "tool": "tool-msg",
        }

        for msg in loaded_messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            tool_calls = msg.get("tool_calls")

            # Clean metadata key before appending to active conversation state
            clean_msg: dict = {"role": role, "content": content}
            if tool_calls:
                clean_msg["tool_calls"] = tool_calls
            self.messages.append(clean_msg)

            if content or role != "assistant":
                self._mount_chat_message(
                    content,
                    role=role,
                    css_class=css_map.get(role, "user-msg"),
                    animate=False,
                )

        # If the last message in the saved session had unresolved tool calls, mount their approval widgets
        if self.messages and self.messages[-1].get("tool_calls"):
            for tc in self.messages[-1]["tool_calls"]:
                approval = CommandApproval(
                    tool_call=tc, tool_registry=self.tool_registry
                )
                self.chat_scroll.mount(approval)

        self._set_view("chat")
        self.notify(f"Loaded session: {session_data.get('title', self.session_id)}")

    @on(Button.Pressed, "#btn-reset")
    async def handle_reset(self) -> None:
        if self.active_worker and self.active_worker.is_running:
            await self.action_cancel_stream()

        self.session_id = self.session_manager.create_session_id()
        self.messages.clear()
        self.sent_images.clear()
        await self.chat_scroll.remove_children()
        self.chat_scroll.mount(
            Static(
                "Chat reset. Attach images and ask a question.",
                id="empty-chat-hint",
            ))
        self.notify("Started a new conversation session.")

    @on(Button.Pressed, "#btn-ask")
    async def action_ask_ollama(self) -> None:
        # If a stream is currently running, pressing the button cancels it
        if self.active_worker and self.active_worker.is_running:
            await self.action_cancel_stream()
            return

        prompt = self._resolve_prompt()
        if not prompt:
            self.notify("Prompt cannot be empty.", severity="error")
            return

        for hint in self.query("#empty-chat-hint"):
            hint.remove()

        self.messages.append(self._build_user_message(prompt))
        self._auto_save_session()

        self.last_user_msg_widget = self._mount_chat_message(
            prompt,
            role="user",
            css_class="user-msg",
            animate=False,
        )
        self._reset_prompt_inputs()
        self._set_view("chat")
        self._trigger_ollama_turn()

    def _trigger_ollama_turn(self) -> None:
        """Sets up the UI to stream an assistant response and fires the worker."""
        pending_approvals = [
            cmd for cmd in self.query(CommandApproval)
            if not cmd.has_class("resolved")
        ]
        if pending_approvals:
            return

        self.streaming_msg_widget = self._mount_chat_message(
            "",
            role="assistant",
            css_class="assistant-msg",
            animate=True,
            streaming=True,
        )
        self._set_streaming_ui_state(True)

        raw_sys_val = self.query_one("#system-select", Select).value
        system_prompt = "" if raw_sys_val == Select.BLANK else str(raw_sys_val)

        ctx_val = self.query_one("#ctx-input", Input).value
        num_ctx = int(
            ctx_val) if ctx_val.isdigit() else self.config.app.default_context
        enable_tools = self.query_one("#tools-toggle", Switch).value

        self.active_worker = self.stream_ollama_response(
            system_prompt=system_prompt,
            num_ctx=num_ctx,
            enable_tools=enable_tools)

    async def _cleanup_stream_ui(self,
                                 append_text: str = "",
                                 notify_msg: str = "",
                                 notify_severity: str = "warning") -> None:
        """DRY helper for cancelling or handling errors mid-stream."""
        self._set_streaming_ui_state(False)
        self.active_worker = None

        if self.streaming_msg_widget is not None:
            partial_text = self.streaming_msg_widget.raw_text.strip()

            if partial_text:
                if append_text:
                    self.streaming_msg_widget.append_chunk(append_text)
                await self.streaming_msg_widget.finalize_stream()
                self.messages.append({
                    "role": "assistant",
                    "content": self.streaming_msg_widget.raw_text.strip()
                })
                self._auto_save_session()
            else:
                self.streaming_msg_widget.remove()
                if self.last_user_msg_widget is not None:
                    self.last_user_msg_widget.remove()
                if self.messages and self.messages[-1]["role"] == "user":
                    popped = self.messages.pop()
                    if "_source_paths" in popped:
                        self.sent_images.difference_update(
                            popped["_source_paths"])
                    self._auto_save_session()

            self.streaming_msg_widget = None
            self.last_user_msg_widget = None

        if notify_msg:
            self.notify(notify_msg, severity=notify_severity)

    async def action_cancel_stream(self) -> None:
        """Cancels an in-progress streaming request."""
        if not self.active_worker or not self.active_worker.is_running:
            return

        self.active_worker.cancel()
        await self._cleanup_stream_ui(
            append_text="\n\n*[Generation cancelled]*",
            notify_msg="Request cancelled.")

    async def handle_stream_error(self, error_msg: str) -> None:
        await self._cleanup_stream_ui(notify_msg=f"API Error: {error_msg}",
                                      notify_severity="error")

    def _set_streaming_ui_state(self, is_streaming: bool) -> None:
        ask_btn = self.query_one("#btn-ask", Button)
        if is_streaming:
            ask_btn.label = "Cancel (Esc)"
            ask_btn.variant = "error"
        else:
            ask_btn.label = "Ask Ollama"
            ask_btn.variant = "success"

    def _resolve_prompt(self) -> str:
        preset_val = self.preset_select.value
        custom_text = self.prompt_input.text.strip()
        if preset_val == "Custom":
            return custom_text
        return custom_text or str(preset_val)

    def _build_user_message(self, prompt: str) -> dict:
        msg_entry: dict = {"role": "user", "content": prompt}
        new_images = [
            p for p in self.workspace.get_attached_images()
            if p not in self.sent_images
        ]

        if new_images:
            msg_entry["images"] = [
                base64.b64encode(p.read_bytes()).decode("utf-8")
                for p in new_images
            ]
            msg_entry["_source_paths"] = new_images
            self.sent_images.update(new_images)

        return msg_entry

    def _mount_chat_message(
        self,
        text: str,
        role: str,
        css_class: str,
        animate: bool = True,
        streaming: bool = False,
    ) -> ChatMessage:
        is_markdown = self.md_enabled
        widget = ChatMessage(
            raw_text=text,
            role=role,
            model_name=self.model if role == "assistant" else "",
            is_markdown=is_markdown,
            streaming=streaming,
            classes=css_class,
        )
        self.chat_scroll.mount(widget)
        self.chat_scroll.scroll_end(animate=animate)
        return widget

    def _reset_prompt_inputs(self) -> None:
        self.prompt_input.clear()
        self.preset_select.value = "Custom"

    @work
    async def stream_ollama_response(self,
                                     system_prompt: str = "",
                                     num_ctx: int = 2048,
                                     enable_tools: bool = True) -> None:
        """Native asyncio task running in the Textual event loop."""
        worker = get_current_worker()

        messages_snapshot = [{
            k: v for k, v in msg.items() if not k.startswith("_")
        } for msg in self.messages]

        if system_prompt.strip():
            messages_snapshot.insert(0, {
                "role": "system",
                "content": system_prompt
            })

        # DRY image-stripped payload saving for the debug modal
        def _strip_images(msg: dict) -> dict:
            d = msg.copy()
            if "images" in d:
                d["images"] = [
                    f"<base64 data, length: {len(b)}>" for b in d["images"]
                ]
            return d

        tools = self.tool_registry.get_schemas() if enable_tools else None

        self.last_api_payload = {
            "model": self.model,
            "messages": [_strip_images(m) for m in messages_snapshot],
            "options": {
                "num_ctx": num_ctx
            },
            "stream": True,
        }

        if tools:
            self.last_api_payload["tools"] = tools

        ollama_options = {"num_ctx": num_ctx}
        collected_tool_calls = []

        async for chunk, error_msg, metrics, tool_calls in stream_ollama_chat_async(
                self.model,
                messages_snapshot,
                self.config,
                options=ollama_options,
                tools=tools):
            if worker.is_cancelled:
                return
            if error_msg:
                await self.handle_stream_error(error_msg)
                return
            if chunk:
                self.handle_stream_chunk(chunk)
            if tool_calls:
                collected_tool_calls.extend(tool_calls)
            if metrics:
                p_count = metrics.get("prompt_eval_count", 0)
                e_count = metrics.get("eval_count", 0)
                total = p_count + e_count
                self.notify(
                    f"Prompt: {p_count} | Response: {e_count} | Total: {total}",
                    title="Token Usage",
                    timeout=5)

        if not worker.is_cancelled:
            await self.handle_stream_complete(tool_calls=collected_tool_calls)

    def handle_stream_chunk(self, chunk: str) -> None:
        if self.streaming_msg_widget is None:
            return
        self.streaming_msg_widget.append_chunk(chunk)

    async def handle_stream_complete(self,
                                     tool_calls: list[dict] | None = None
                                     ) -> None:
        self._set_streaming_ui_state(False)
        self.active_worker = None

        if self.streaming_msg_widget is None:
            return

        final_content = self.streaming_msg_widget.raw_text
        if not final_content.strip() and not tool_calls:
            self.streaming_msg_widget.remove()
            self.streaming_msg_widget = None
            self.notify("Received empty response from the model.",
                        severity="error")
            return
        elif not final_content.strip() and tool_calls:
            self.streaming_msg_widget.remove()
        else:
            await self.streaming_msg_widget.finalize_stream()

        assistant_msg: dict = {"role": "assistant", "content": final_content}
        if tool_calls:
            assistant_msg["tool_calls"] = tool_calls

        self.messages.append(assistant_msg)
        self._auto_save_session()

        self.streaming_msg_widget = None
        self.last_user_msg_widget = None

        if tool_calls:
            for tc in tool_calls:
                approval = CommandApproval(tool_call=tc, tool_registry=self.tool_registry)
                self.chat_scroll.mount(approval)

        self.chat_scroll.scroll_end(animate=True)

    def _get_approval_widget(self, button: Button) -> CommandApproval | None:
        """Helper to find the parent CommandApproval widget from a Button."""
        for node in button.ancestors:
            if isinstance(node, CommandApproval):
                return node
        return None

    def _resolve_command(self, approval_widget: CommandApproval,
                         internal_content: str, display_text: str) -> None:
        """DRY helper for updating UI state after a user interacts with a command approval block."""
        approval_widget.mark_resolved()
        self.messages.append({
            "role": "tool",
            "content": internal_content,
        })
        self._auto_save_session()

        self._mount_chat_message(
            display_text,
            role="tool",
            css_class="tool-msg",
            animate=False,
        )
        approval_widget.remove()
        self._trigger_ollama_turn()

    @on(Button.Pressed, "#btn-accept-cmd")
    async def on_accept_cmd(self, event: Button.Pressed) -> None:
        approval_widget = self._get_approval_widget(event.button)
        if not approval_widget or approval_widget.has_class("resolved"):
            return

        tool_name = approval_widget.tool_name
        args = approval_widget.args

        tool = self.tool_registry.get_tool(tool_name)
        if not tool:
            self._resolve_command(
                approval_widget,
                f"Tool '{tool_name}' not found in registry.",
                f"**Error:** Tool `{tool_name}` not registered."
            )
            return

        try:
            internal_content, display_text = await tool.execute(args)
        except Exception as e:
            internal_content = f"Execution crashed: {e}"
            display_text = f"**Tool Crashed:**\n```\n{e}\n```"

        feedback_input = approval_widget.query_one("#cmd-feedback-input", Input)
        feedback = feedback_input.value.strip() if feedback_input else ""
        if feedback:
            internal_content += f"\n\n[User attached a message]: {feedback}"
            display_text += f"\n\n**User Message:** {feedback}"

        self._resolve_command(approval_widget, internal_content, display_text)

    @on(Button.Pressed, "#btn-decline-cmd")
    def on_decline_cmd(self, event: Button.Pressed) -> None:
        approval_widget = self._get_approval_widget(event.button)
        if not approval_widget or approval_widget.has_class("resolved"):
            return

        feedback_input = approval_widget.query_one("#cmd-feedback-input", Input)
        feedback = feedback_input.value.strip() if feedback_input else ""

        internal_msg = "The user declined to run this command."
        display_msg = "*Command execution declined by user.*"
        
        if feedback:
            internal_msg += f" User provided this reason/message: {feedback}"
            display_msg += f"\n\n**User Message:** {feedback}"

        self._resolve_command(approval_widget, internal_msg, display_msg)

    def action_toggle_view(self) -> None:
        """Toggles between the prep panel and the chat panel."""
        self._set_view("chat" if self.view == "prep" else "prep")

    def action_show_debug(self) -> None:
        """Shows the raw API payload of the last request."""
        if not self.last_api_payload:
            self.notify("No requests have been sent yet.", severity="warning")
            return
        self.push_screen(DebugModal(self.last_api_payload))

    def cleanup(self) -> None:
        self.workspace.cleanup()
