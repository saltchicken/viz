import base64
from pathlib import Path

from textual import on
from textual import work
from textual.app import App
from textual.app import ComposeResult
from textual.containers import Horizontal
from textual.containers import Vertical
from textual.containers import VerticalScroll
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

from viz.config import config
from viz.modals import DebugModal
from viz.modals import FileSelectModal
from viz.services import ImageWorkspace
from viz.services import stream_ollama_chat_async
from viz.widgets import ChatMessage

PRESETS = [
    ("Custom", "Custom"),
    ("Describe this image in detail.", "Describe this image in detail."),
    (
        "Extract all the text from this image.",
        "Extract all the text from this image.",
    ),
    (
        "Create FLUX3 Prompt",
        "Create a comprehensive, highly descriptive prompt optimized for the FLUX3 text-to-image model. Include the main subject, artistic style, camera angle, lighting, textures, and mood. Output ONLY the prompt.",
    ),
]


class OllamaVisionApp(App):
    CSS_PATH = "styles.tcss"

    BINDINGS = [
        ("ctrl+enter", "ask_ollama", "Ask / Send"),
        ("ctrl+o", "toggle_layout", "Toggle Layout"),
        ("ctrl+1", "show_debug", "Debug API"),
        ("ctrl+q", "quit", "Quit"),
    ]

    def __init__(self, model: str) -> None:
        super().__init__()
        self.model = model
        self.title = f"Ollama Vision ({self.model})"
        self.workspace = ImageWorkspace()

        self.messages: list[dict] = []
        self.sent_images: set[Path] = set()
        self.active_worker: Worker | None = None
        self.streaming_msg_widget: ChatMessage | None = None
        self.last_user_msg_widget: ChatMessage | None = None
        self.last_api_payload: dict = {}

        # 0: Split View, 1: Prep Panel Only, 2: Chat Panel Only
        self.layout_mode = 0

    def compose(self) -> ComposeResult:
        yield Header()
        with Horizontal(id="main-split"):
            yield from self._compose_prep_panel()
            yield from self._compose_chat_panel()
        yield Footer()

    def _compose_prep_panel(self) -> ComposeResult:
        with Vertical(id="prep-panel"):
            yield Label("Prepare Vision Prompt", classes="section-title")
            yield Static("No Images Captured", id="preview-state")
            yield TextualImage(id="preview-image")

            with Horizontal(classes="btn-row"):
                yield Button("Capture", id="btn-capture", variant="primary")
                yield Button("Paste", id="btn-paste")
            with Horizontal(classes="btn-row"):
                yield Button("Add File", id="btn-add-file")
                yield Button("Clear", id="btn-clear", variant="warning")
                
            yield Label("System Message:")
            yield Select(
                config.system_presets,
                id="system-select",
                allow_blank=True,
            )

            yield Label("Select Preset:")
            yield Select(
                config.presets,
                value="Custom",
                id="preset-select",
                allow_blank=False,
            )
            
            yield Label("Context Size:")
            yield Input(value="32768", type="integer", id="ctx-input")

            yield Label("Prompt (Multiline):", id="prompt-label")
            yield TextArea(id="prompt-input")

            with Horizontal(classes="btn-row"):
                yield Button(
                    "Ask Ollama (Ctrl+S)",
                    id="btn-ask",
                    variant="success",
                )
                yield Button("Reset Chat", id="btn-reset", variant="error")

    def _compose_chat_panel(self) -> ComposeResult:
        with Vertical(id="chat-panel"):
            with Horizontal(id="chat-header"):
                yield Label(
                    "Conversation History",
                    id="chat-header-title",
                    classes="section-title",
                )
                with Horizontal(id="toggle-container"):
                    yield Label("Raw")
                    yield Switch(value=True, id="view-toggle")
                    yield Label("MD")

            with VerticalScroll(id="chat-scroll"):
                yield Static(
                    "Attach images and enter a prompt on the left to start chatting.",
                    id="empty-chat-hint",
                )

    def on_mount(self) -> None:
        self.update_preview()
        self.query_one("#prompt-input", TextArea).focus()

    @on(Switch.Changed, "#view-toggle")
    def on_view_toggle(self, event: Switch.Changed) -> None:
        for msg in self.query(ChatMessage):
            msg.toggle_view(event.value)

    def update_preview(self) -> None:
        png_files = self.workspace.get_attached_images()
        count = len(png_files)
        state_label = self.query_one("#preview-state", Static)
        img_widget = self.query_one("#preview-image", TextualImage)

        if count == 0:
            state_label.update(
                "Images Attached: [b]0[/b] — Click 'Capture', 'Paste', or 'Add File'"
            )
            img_widget.image = None
            img_widget.display = False
            self.sub_title = "0 images attached"
            return

        state_label.update(f"Images Attached: [b]{count}[/b]")
        img_widget.display = True
        self.sub_title = f"{count} image(s) attached"
        img_widget.image = self.workspace.build_preview()

    @on(Select.Changed, "#preset-select")
    def on_preset_changed(self, event: Select.Changed) -> None:
        prompt_area = self.query_one("#prompt-input", TextArea)
        if event.value == "Custom":
            prompt_area.disabled = False
        else:
            prompt_area.text = str(event.value)

    @on(Button.Pressed, "#btn-capture")
    def handle_capture(self) -> None:
        with self.suspend():
            captured = self.workspace.capture_screen_region()

        if captured:
            self.notify("Screen region captured.")
        else:
            self.notify("Capture cancelled.", severity="warning")
        self.update_preview()

    @on(Button.Pressed, "#btn-paste")
    def handle_paste(self) -> None:
        if self.workspace.paste_clipboard_image():
            self.notify("Image pasted from clipboard.")
            self.update_preview()
        else:
            self.notify("No image found in clipboard.", severity="error")

    @on(Button.Pressed, "#btn-add-file")
    def handle_add_file(self) -> None:
        self.push_screen(FileSelectModal(), self._on_files_selected)

    def _on_files_selected(self, files: list[Path] | None) -> None:
        if not files:
            return
        added = self.workspace.import_files(files)
        if added:
            self.notify(f"Added {added} image(s).")
        else:
            self.notify("Failed to convert selected file(s).", severity="error")
        self.update_preview()

    @on(Button.Pressed, "#btn-clear")
    def handle_clear(self) -> None:
        self.workspace.clear_images()
        self.update_preview()
        self.notify("Cleared attached images.")

    @on(Button.Pressed, "#btn-reset")
    def handle_reset(self) -> None:
        if self.active_worker and self.active_worker.is_running:
            self.action_cancel_stream()

        self.messages.clear()
        self.sent_images.clear()
        chat_scroll = self.query_one("#chat-scroll", VerticalScroll)
        chat_scroll.remove_children()
        chat_scroll.mount(
            Static(
                "Chat reset. Attach images and ask a question.",
                id="empty-chat-hint",
            ))
        self.query_one("#prompt-label", Label).update("Prompt (Multiline):")
        self.notify("Conversation history cleared.")

    @on(Button.Pressed, "#btn-ask")
    def action_ask_ollama(self) -> None:
        # If a stream is currently running, pressing the button cancels it
        if self.active_worker and self.active_worker.is_running:
            self.action_cancel_stream()
            return

        prompt = self._resolve_prompt()
        if not prompt:
            self.notify("Prompt cannot be empty.", severity="error")
            return

        for hint in self.query("#empty-chat-hint"):
            hint.remove()
            
        raw_sys_val = self.query_one("#system-select", Select).value
        system_prompt = "" if raw_sys_val == Select.BLANK else str(raw_sys_val)

        ctx_val = self.query_one("#ctx-input", Input).value
        num_ctx = int(ctx_val) if ctx_val.isdigit() else 2048

        self.messages.append(self._build_user_message(prompt))
        self.last_user_msg_widget = self._mount_chat_message(
            prompt,
            role="user",
            css_class="user-msg",
            animate=False,
        )
        self._reset_prompt_inputs()

        self.streaming_msg_widget = self._mount_chat_message(
            "",
            role="assistant",
            css_class="assistant-msg",
            animate=True,
            streaming=True,
        )
        self._set_streaming_ui_state(True)
        self.active_worker = self.stream_ollama_response(
            system_prompt=system_prompt,
            num_ctx=num_ctx
        )

    def action_cancel_stream(self) -> None:
        """Cancels an in-progress streaming request."""
        if not self.active_worker or not self.active_worker.is_running:
            return

        self.active_worker.cancel()
        self.active_worker = None
        self._set_streaming_ui_state(False)

        if self.streaming_msg_widget is not None:
            partial_text = self.streaming_msg_widget.raw_text.strip()
            if partial_text:
                # Keep the partial response in history so context isn't lost
                self.streaming_msg_widget.append_chunk(
                    "\n\n*[Generation cancelled]*")
                self.streaming_msg_widget.finalize_stream()
                self.messages.append({
                    "role": "assistant",
                    "content": partial_text
                })
            else:
                # Nothing arrived yet: roll back the empty assistant card and user message
                self.streaming_msg_widget.remove()
                if self.last_user_msg_widget is not None:
                    self.last_user_msg_widget.remove()
                if self.messages and self.messages[-1]["role"] == "user":
                    popped = self.messages.pop()
                    if "_source_paths" in popped:
                        self.sent_images.difference_update(popped["_source_paths"])
                        
                    if not any("images" in m for m in self.messages):
                        self.query_one("#prompt-label",
                                       Label).update("Prompt (Multiline):")

            self.streaming_msg_widget = None
            self.last_user_msg_widget = None

        self.notify("Request cancelled.", severity="warning")

    def _set_streaming_ui_state(self, is_streaming: bool) -> None:
        ask_btn = self.query_one("#btn-ask", Button)
        if is_streaming:
            ask_btn.label = "Cancel (Esc)"
            ask_btn.variant = "error"
        else:
            ask_btn.label = "Ask Ollama (Ctrl+S)"
            ask_btn.variant = "success"

    def _resolve_prompt(self) -> str:
        preset_val = self.query_one("#preset-select", Select).value
        custom_text = self.query_one("#prompt-input", TextArea).text.strip()
        if preset_val == "Custom":
            return custom_text
        return custom_text or str(preset_val)

    def _build_user_message(self, prompt: str) -> dict:
        msg_entry: dict = {"role": "user", "content": prompt}
        
        # Find images in the workspace that haven't been sent yet
        new_images = [
            p for p in self.workspace.get_attached_images() 
            if p not in self.sent_images
        ]
        
        if new_images:
            msg_entry["images"] = [
                base64.b64encode(p.read_bytes()).decode("utf-8")
                for p in new_images
            ]
            msg_entry["_source_paths"] = new_images  # Private tracking key
            self.sent_images.update(new_images)
            self.query_one("#prompt-label", Label).update("Follow-up Question:")
            
        return msg_entry

    def _mount_chat_message(
        self,
        text: str,
        role: str,
        css_class: str,
        animate: bool = True,
        streaming: bool = False,
    ) -> ChatMessage:
        is_markdown = self.query_one("#view-toggle", Switch).value
        chat_scroll = self.query_one("#chat-scroll", VerticalScroll)
        widget = ChatMessage(
            raw_text=text,
            role=role,
            model_name=self.model if role == "assistant" else "",
            is_markdown=is_markdown,
            streaming=streaming,
            classes=css_class,
        )
        chat_scroll.mount(widget)
        chat_scroll.scroll_end(animate=animate)
        return widget

    def _reset_prompt_inputs(self) -> None:
        self.query_one("#prompt-input", TextArea).clear()
        self.query_one("#preset-select", Select).value = "Custom"

    @work
    async def stream_ollama_response(self, system_prompt: str = "", num_ctx: int = 2048) -> None:
        """Native asyncio task running in the Textual event loop."""
        worker = get_current_worker()
        
        # Filter out internal tracking keys (like '_source_paths') 
        # before passing the snapshot to the Ollama API
        messages_snapshot = [
            {k: v for k, v in msg.items() if not k.startswith("_")}
            for msg in self.messages
        ]

        if system_prompt.strip():
            messages_snapshot.insert(0, {"role": "system", "content": system_prompt})
            
        # Prepare the truncated debug payload so it doesn't freeze the UI rendering huge base64 strings
        debug_messages = []
        for msg in messages_snapshot:
            debug_msg = {k: v for k, v in msg.items() if k != "images"}
            if "images" in msg:
                debug_msg["images"] = [f"<base64 data, length: {len(b64)}>" for b64 in msg["images"]]
            debug_messages.append(debug_msg)
            
        self.last_api_payload = {
            "model": self.model,
            "messages": debug_messages,
            "options": {"num_ctx": num_ctx},
            "stream": True,
        }

        ollama_options = {"num_ctx": num_ctx}

        async for chunk, error_msg, metrics in stream_ollama_chat_async(
            self.model, 
            messages_snapshot, 
            options=ollama_options
        ):
            if worker.is_cancelled:
                return
            if error_msg:
                self.handle_stream_error(error_msg)
                return
            if chunk:
                self.handle_stream_chunk(chunk)
            if metrics:
                p_count = metrics.get("prompt_eval_count", 0)
                e_count = metrics.get("eval_count", 0)
                total = p_count + e_count
                self.notify(
                    f"Prompt: {p_count} | Response: {e_count} | Total: {total}", 
                    title="Token Usage", 
                    timeout=5
                )

        if not worker.is_cancelled:
            self.handle_stream_complete()

    def handle_stream_chunk(self, chunk: str) -> None:
        if self.streaming_msg_widget is None:
            return
        self.streaming_msg_widget.append_chunk(chunk)
        chat_scroll = self.query_one("#chat-scroll", VerticalScroll)
        chat_scroll.scroll_end(animate=False)

    def handle_stream_error(self, error_msg: str) -> None:
        self._set_streaming_ui_state(False)
        self.active_worker = None

        if self.streaming_msg_widget is not None:
            if not self.streaming_msg_widget.raw_text.strip():
                self.streaming_msg_widget.remove()
            else:
                self.streaming_msg_widget.finalize_stream()
            self.streaming_msg_widget = None

        self.last_user_msg_widget = None
        self.notify(f"API Error: {error_msg}", severity="error", timeout=8)

    def handle_stream_complete(self) -> None:
        self._set_streaming_ui_state(False)
        self.active_worker = None

        if self.streaming_msg_widget is None:
            return

        final_content = self.streaming_msg_widget.raw_text
        if not final_content.strip():
            self.streaming_msg_widget.remove()
            self.streaming_msg_widget = None
            self.notify("Received empty response from the model.",
                        severity="error")
            return

        self.streaming_msg_widget.finalize_stream()
        self.messages.append({"role": "assistant", "content": final_content})
        self.streaming_msg_widget = None
        self.last_user_msg_widget = None

        chat_scroll = self.query_one("#chat-scroll", VerticalScroll)
        chat_scroll.scroll_end(animate=True)

    def action_toggle_layout(self) -> None:
        """Cycles through 3 layout modes: Split -> Prep Only -> Chat Only."""
        self.layout_mode = (self.layout_mode + 1) % 3

        prep_panel = self.query_one("#prep-panel", Vertical)
        chat_panel = self.query_one("#chat-panel", Vertical)

        if self.layout_mode == 0:
            prep_panel.display = True
            chat_panel.display = True
            prep_panel.styles.width = None
            chat_panel.styles.width = None
            self.notify("Layout: Split View")

        elif self.layout_mode == 1:
            prep_panel.display = True
            chat_panel.display = False
            prep_panel.styles.width = "100%"
            self.notify("Layout: Prep Panel Only")

        else:
            prep_panel.display = False
            chat_panel.display = True
            chat_panel.styles.width = "100%"
            self.notify("Layout: Chat Panel Only")
            
    def action_show_debug(self) -> None:
        """Shows the raw API payload of the last request."""
        if not self.last_api_payload:
            self.notify("No requests have been sent yet.", severity="warning")
            return
        self.push_screen(DebugModal(self.last_api_payload))

    def cleanup(self) -> None:
        self.workspace.cleanup()
