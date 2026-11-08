#!/usr/bin/env python3
import base64
from datetime import datetime
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile

import requests
from textual import on
from textual import work
from textual.app import App
from textual.app import ComposeResult
from textual.containers import Horizontal
from textual.containers import Vertical
from textual.containers import VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Button
from textual.widgets import DirectoryTree
from textual.widgets import Footer
from textual.widgets import Header
from textual.widgets import Label
from textual.widgets import Markdown
from textual.widgets import Select
from textual.widgets import Static
from textual.widgets import Switch
from textual.widgets import TextArea
from textual.worker import get_current_worker
# Use textual-image for native Kitty/Sixel terminal pixel rendering
from textual_image.widget import Image as TextualImage

PRESETS = [
    ("Custom", "Custom"),
    ("Describe this image in detail.", "Describe this image in detail."),
    ("Extract all the text from this image.",
     "Extract all the text from this image."),
    ("Create FLUX3 Prompt",
     "Create a comprehensive, highly descriptive prompt optimized for the FLUX3 text-to-image model. Include the main subject, artistic style, camera angle, lighting, textures, and mood. Output ONLY the prompt."
    ),
]


class FileSelectModal(ModalScreen[list[Path]]):
    """Modal dialog to browse and select image files."""
    CSS = """
    FileSelectModal {
        align: center middle;
    }
    #modal-dialog {
        width: 80%;
        height: 80%;
        border: thick $primary;
        background: $surface;
        padding: 1 2;
    }
    #file-tree {
        height: 1fr;
        border: solid $secondary;
        margin: 1 0;
    }
    #selected-label {
        height: 3;
        color: $accent;
    }
    .modal-buttons {
        height: auto;
        align: right middle;
    }
    Button {
        margin-left: 1;
    }
    """

    def __init__(self):
        super().__init__()
        self.selected_files: list[Path] = []

    def compose(self) -> ComposeResult:
        with Vertical(id="modal-dialog"):
            yield Label(
                "[b]Select Image(s)[/b] (Click files to toggle selection)")
            yield DirectoryTree(Path.home(), id="file-tree")
            yield Static("Selected: 0 files", id="selected-label")
            with Horizontal(classes="modal-buttons"):
                yield Button("Cancel", id="cancel-btn", variant="error")
                yield Button("Add Selected", id="add-btn", variant="success")

    @on(DirectoryTree.FileSelected, "#file-tree")
    def on_file_selected(self, event: DirectoryTree.FileSelected) -> None:
        path = event.path
        if path in self.selected_files:
            self.selected_files.remove(path)
        else:
            self.selected_files.append(path)
        names = ", ".join(p.name for p in self.selected_files)
        self.query_one(
            "#selected-label",
            Static).update(f"Selected ({len(self.selected_files)}): {names}")

    @on(Button.Pressed, "#cancel-btn")
    def on_cancel(self) -> None:
        self.dismiss([])

    @on(Button.Pressed, "#add-btn")
    def on_add(self) -> None:
        self.dismiss(self.selected_files)


class ThinkingModal(ModalScreen[bool]):
    """Modal shown while waiting for Ollama, supporting request cancellation."""
    CSS = """
    ThinkingModal {
        align: center middle;
    }
    #thinking-box {
        width: 60;
        height: auto;
        border: heavy $accent;
        background: $surface;
        padding: 2;
        align: center middle;
    }
    #thinking-text {
        text-align: center;
        margin-bottom: 2;
    }
    """

    def __init__(self, prompt_snippet: str):
        super().__init__()
        self.prompt_snippet = prompt_snippet

    def compose(self) -> ComposeResult:
        with Vertical(id="thinking-box"):
            yield Static(
                f"[b]Thinking...[/b]\n\n"
                f"Please wait while the model generates a response.\n\n"
                f"[i]'{self.prompt_snippet[:50]}...'[/i]",
                id="thinking-text")
            yield Button("Cancel Request", id="cancel-req-btn", variant="error")

    @on(Button.Pressed, "#cancel-req-btn")
    def cancel_request(self) -> None:
        self.dismiss(False)


class CopyCodeButton(Button):
    """A button that holds the specific code block text to be copied."""
    def __init__(self, code_text: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.code_text = code_text


class ChatMessage(Vertical):
    """A container holding both Markdown and Raw text views."""
    
    def __init__(self, raw_text: str, role: str, model_name: str = "", is_markdown: bool = True, **kwargs):
        super().__init__(**kwargs)
        self.raw_text = raw_text
        self.role = role
        self.model_name = model_name
        self.is_markdown = is_markdown

    def compose(self) -> ComposeResult:
        prefix_md = "**You:**\n\n" if self.role == "user" else f"**Ollama ({self.model_name}):**\n\n"
        prefix_raw = "You:\n\n" if self.role == "user" else f"Ollama ({self.model_name}):\n\n"
        
        # Markdown View Container (Split into chunks and code blocks)
        with Vertical(classes="md-view-container"):
            chunks = re.split(r'(```.*?```)', self.raw_text, flags=re.DOTALL)
            if chunks:
                chunks[0] = prefix_md + chunks[0]
            
            code_idx = 1
            for chunk in chunks:
                if not chunk.strip():
                    continue
                
                if chunk.startswith("```") and chunk.endswith("```"):
                    match = re.match(r'```[^\n]*\n(.*?)```', chunk, re.DOTALL)
                    inner_code = match.group(1).strip() if match else chunk.strip('`').strip()
                    
                    yield Markdown(chunk, classes="md-chunk")
                    yield CopyCodeButton(
                        inner_code, 
                        label=f"📋", 
                        classes="copy-code-btn inline-copy-btn",
                        variant="primary"
                    )
                    code_idx += 1
                else:
                    yield Markdown(chunk, classes="md-chunk")

        # Raw View Container
        self.raw_view = Static(f"{prefix_raw}{self.raw_text}")
        yield self.raw_view

    def on_mount(self) -> None:
        self.query_one(".md-view-container", Vertical).display = self.is_markdown
        self.raw_view.display = not self.is_markdown

    def toggle_view(self, show_markdown: bool) -> None:
        self.query_one(".md-view-container", Vertical).display = show_markdown
        self.raw_view.display = not show_markdown

    @on(Button.Pressed, ".copy-code-btn")
    def on_copy_code(self, event: Button.Pressed) -> None:
        button = event.button
        assert isinstance(button, CopyCodeButton)
        
        try:
            self.app.copy_to_clipboard(button.code_text)
            self.app.notify(f"Copied {button.label} to clipboard!")
        except Exception:
            try:
                subprocess.run(["wl-copy"], input=button.code_text.encode("utf-8"), check=True)
                self.app.notify(f"Copied {button.label} via wl-copy!")
            except Exception as e:
                self.app.notify(f"Clipboard error: {e}", severity="error")


class OllamaVisionApp(App):
    CSS = """
    #main-split {
        height: 1fr;
    }
    #prep-panel {
        width: 42%;
        height: 1fr;
        border-right: solid $primary;
        padding: 1;
    }
    #chat-panel {
        width: 58%;
        height: 1fr;
        padding: 1;
    }
    .section-title {
        text-style: bold;
        color: $accent;
        margin-bottom: 1;
    }
    #chat-header {
        height: 3;
        margin-bottom: 1;
    }
    #chat-header-title {
        width: 1fr;
        content-align: left middle;
    }
    #toggle-container {
        width: auto;
        height: 3;
        align: right middle;
    }
    #toggle-container Label {
        margin-top: 1;
        margin-left: 1;
        margin-right: 1;
    }
    #preview-state {
        height: auto;
        color: $text-muted;
        margin-bottom: 1;
    }
    #preview-image {
        height: 14;
        width: auto;
        border: dashed $secondary;
        margin-bottom: 1;
    }
    .btn-row {
        height: auto;
        margin-bottom: 1;
    }
    .btn-row Button {
        width: 1fr;
        margin-right: 1;
    }
    #prompt-input {
        height: 1fr;
        min-height: 6;
        margin-bottom: 1;
    }
    #chat-scroll {
        height: 1fr;
        border: solid $secondary;
        padding: 1;
        margin-bottom: 1;
    }
    ChatMessage {
        height: auto;
        padding: 1;
        margin-bottom: 1;
    }
    .user-msg {
        background: $boost;
        border-left: thick $primary;
    }
    .assistant-msg {
        background: $surface;
        border-left: thick $success;
    }
    .md-view-container {
        height: auto;
    }
    .md-chunk {
        height: auto;
        margin-bottom: 0;
    }
    .inline-copy-btn {
        min-width: 4;
        margin-top: 0;
        margin-bottom: 1;
        border: none;
        height: 1;
        align: right middle;
    }
    """

    BINDINGS = [
        ("q", "quit", "Quit"),
        ("ctrl+s", "ask_ollama", "Ask / Send"),
    ]

    def __init__(self, model: str):
        super().__init__()
        self.model = model
        self.title = f"Ollama Vision ({self.model})"
        self.tmp_dir = Path(tempfile.mkdtemp(prefix="ollama-vision-"))
        self.img_dir = self.tmp_dir / "images"
        self.img_dir.mkdir(parents=True, exist_ok=True)

        self.messages: list[dict] = []
        self.first_turn = True
        self.active_worker = None
        self.thinking_modal: ThinkingModal | None = None

    def compose(self) -> ComposeResult:
        yield Header()
        with Horizontal(id="main-split"):
            # Left Panel: Image Preparation & Prompt Setup
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

                yield Label("Select Preset:")
                yield Select(PRESETS,
                             value="Custom",
                             id="preset-select",
                             allow_blank=False)

                yield Label("Prompt (Multiline):", id="prompt-label")
                yield TextArea(id="prompt-input")

                with Horizontal(classes="btn-row"):
                    yield Button("Ask Ollama (Ctrl+S)",
                                 id="btn-ask",
                                 variant="success")
                    yield Button("Reset Chat", id="btn-reset", variant="error")

            # Right Panel: Markdown Chat History
            with Vertical(id="chat-panel"):
                with Horizontal(id="chat-header"):
                    yield Label("Conversation History", id="chat-header-title", classes="section-title")
                    with Horizontal(id="toggle-container"):
                        yield Label("Raw")
                        yield Switch(value=True, id="view-toggle")
                        yield Label("MD")
                
                with VerticalScroll(id="chat-scroll"):
                    yield Static(
                        "Attach images and enter a prompt on the left to start chatting.",
                        id="empty-chat-hint")
        yield Footer()

    def on_mount(self) -> None:
        self.update_preview()

        self.query_one("#prompt-input", TextArea).focus()

    @on(Switch.Changed, "#view-toggle")
    def on_view_toggle(self, event: Switch.Changed) -> None:
        """Toggle Markdown and Raw views across all rendered messages."""
        show_markdown = event.value
        for msg in self.query(ChatMessage):
            msg.toggle_view(show_markdown)

    def get_attached_images(self) -> list[Path]:
        return sorted(self.img_dir.glob("*.png"),
                      key=lambda p: p.stat().st_mtime)

    def update_preview(self) -> None:
        png_files = self.get_attached_images()
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

        preview_path = self.tmp_dir / "preview.png"

        if count == 1:
            img_widget.image = png_files[0]
        else:
            cmd = ["magick", "montage"
                  ] if shutil.which("magick") else ["montage"]
            subprocess.run([
                *cmd,
                "-background",
                "#1e1e2e",
                "-geometry",
                "400x400>+8+8",
                *[str(p) for p in png_files],
                str(preview_path),
            ],
                           capture_output=True)
            img_widget.image = preview_path

    def im_convert(self, src: Path, dst: Path) -> bool:
        cmd = "magick" if shutil.which("magick") else "convert"
        res = subprocess.run([cmd, str(src), str(dst)], capture_output=True)
        return res.returncode == 0

    @on(Select.Changed, "#preset-select")
    def on_preset_changed(self, event: Select.Changed) -> None:
        prompt_area = self.query_one("#prompt-input", TextArea)
        if event.value == "Custom":
            prompt_area.disabled = False
        else:
            prompt_area.text = str(event.value)

    @on(Button.Pressed, "#btn-capture")
    def handle_capture(self) -> None:
        timestamp = datetime.now().strftime("%s%f")
        img_file = self.img_dir / f"shot_{timestamp}.png"

        with self.suspend():
            slurp = subprocess.run(["slurp"], capture_output=True, text=True)
            if slurp.returncode == 0 and slurp.stdout.strip():
                geom = slurp.stdout.strip()
                subprocess.run(["grim", "-g", geom, str(img_file)])

        if img_file.exists() and img_file.stat().st_size > 0:
            self.notify("Screen region captured.")
        else:
            img_file.unlink(missing_ok=True)
            self.notify("Capture cancelled.", severity="warning")

        self.update_preview()

    @on(Button.Pressed, "#btn-paste")
    def handle_paste(self) -> None:
        timestamp = datetime.now().strftime("%s%f")
        img_file = self.img_dir / f"paste_{timestamp}.png"
        with open(img_file, "wb") as f:
            subprocess.run(["wl-paste", "-t", "image/png"],
                           stdout=f,
                           stderr=subprocess.DEVNULL)

        if not img_file.exists() or img_file.stat().st_size == 0:
            img_file.unlink(missing_ok=True)
            self.notify("No image found in clipboard.", severity="error")
        else:
            self.notify("Image pasted from clipboard.")
            self.update_preview()

    @on(Button.Pressed, "#btn-add-file")
    def handle_add_file(self) -> None:

        def process_files(files: list[Path] | None) -> None:
            if not files:
                return
            added = 0
            for f in files:
                if f.is_file():
                    timestamp = datetime.now().strftime("%s%f")
                    out = self.img_dir / f"file_{timestamp}_{added}.png"
                    if self.im_convert(f, out):
                        added += 1
            if added:
                self.notify(f"Added {added} image(s).")
            else:
                self.notify("Failed to convert selected file(s).",
                            severity="error")
            self.update_preview()

        self.push_screen(FileSelectModal(), process_files)

    @on(Button.Pressed, "#btn-clear")
    def handle_clear(self) -> None:
        for p in self.img_dir.glob("*.png"):
            p.unlink(missing_ok=True)
        self.update_preview()
        self.notify("Cleared attached images.")

    @on(Button.Pressed, "#btn-reset")
    def handle_reset(self) -> None:
        self.messages.clear()
        self.first_turn = True
        chat_scroll = self.query_one("#chat-scroll", VerticalScroll)
        chat_scroll.remove_children()
        chat_scroll.mount(
            Static("Chat reset. Attach images and ask a question.",
                   id="empty-chat-hint"))
        self.query_one("#prompt-label", Label).update("Prompt (Multiline):")
        self.notify("Conversation history cleared.")

    @on(Button.Pressed, "#btn-ask")
    def action_ask_ollama(self) -> None:
        preset_val = self.query_one("#preset-select", Select).value
        custom_text = self.query_one("#prompt-input", TextArea).text.strip()

        prompt = custom_text if preset_val == "Custom" else (custom_text or
                                                             str(preset_val))
        if not prompt:
            self.notify("Prompt cannot be empty.", severity="error")
            return

        for hint in self.query("#empty-chat-hint"):
            hint.remove()

        msg_entry: dict = {"role": "user", "content": prompt}
        if self.first_turn:
            png_files = self.get_attached_images()
            if png_files:
                b64_images = [
                    base64.b64encode(p.read_bytes()).decode("utf-8")
                    for p in png_files
                ]
                msg_entry["images"] = b64_images
            self.first_turn = False
            self.query_one("#prompt-label", Label).update("Follow-up Question:")

        self.messages.append(msg_entry)
        
        is_markdown = self.query_one("#view-toggle", Switch).value
        chat_scroll = self.query_one("#chat-scroll", VerticalScroll)
        user_md = ChatMessage(raw_text=prompt, role="user", is_markdown=is_markdown, classes="user-msg")
        chat_scroll.mount(user_md)
        chat_scroll.scroll_end(animate=False)

        self.query_one("#prompt-input", TextArea).clear()
        self.query_one("#preset-select", Select).value = "Custom"

        self.thinking_modal = ThinkingModal(prompt)

        def on_modal_close(completed_normally: bool | None) -> None:
            if completed_normally is False and self.active_worker:
                self.active_worker.cancel()
                if self.messages and self.messages[-1]["role"] == "user":
                    self.messages.pop()
                user_md.remove()
                self.notify("Request cancelled.", severity="warning")

        self.push_screen(self.thinking_modal, on_modal_close)
        self.active_worker = self.fetch_ollama_response()

    @work(thread=True)
    def fetch_ollama_response(self) -> None:
        worker = get_current_worker()
        payload = {
            "model": self.model,
            "messages": self.messages,
            "stream": False,
        }
        try:
            resp = requests.post("http://localhost:11434/api/chat",
                                 json=payload,
                                 timeout=180)
            if worker.is_cancelled:
                return
            data = resp.json()
            error_msg = data.get("error")
            content = data.get("message", {}).get("content", "")
        except Exception as e:
            if worker.is_cancelled:
                return
            error_msg = f"No response or connection timed out: {e}"
            content = ""

        if not worker.is_cancelled:
            self.call_from_thread(self.handle_api_result, content, error_msg)

    def handle_api_result(self, content: str, error_msg: str | None) -> None:
        if self.thinking_modal:
            self.thinking_modal.dismiss(True)
            self.thinking_modal = None

        if error_msg:
            self.notify(f"API Error: {error_msg}", severity="error", timeout=8)
            return

        if not content:
            self.notify("Received empty response from the model.",
                        severity="error")
            return

        self.messages.append({"role": "assistant", "content": content})

        is_markdown = self.query_one("#view-toggle", Switch).value
        chat_scroll = self.query_one("#chat-scroll", VerticalScroll)
        assistant_md = ChatMessage(raw_text=content, role="assistant", model_name=self.model, is_markdown=is_markdown, classes="assistant-msg")
        chat_scroll.mount(assistant_md)
        chat_scroll.scroll_end(animate=True)
        self.notify("Response received.")

    def cleanup(self) -> None:
        shutil.rmtree(self.tmp_dir, ignore_errors=True)


def run_preflight_checks(model: str) -> None:
    required_cmds = ["grim", "slurp", "wl-paste"]
    for cmd in required_cmds:
        if not shutil.which(cmd):
            sys.exit(
                f"[Missing Dependency] Required command '{cmd}' is not installed."
            )

    if not shutil.which("magick") and not shutil.which("convert"):
        sys.exit(
            "[Missing Dependency] ImageMagick is required. Please install 'imagemagick'."
        )

    try:
        r = requests.get("http://localhost:11434", timeout=2)
        r.raise_for_status()
    except Exception:
        sys.exit(
            "[Ollama Not Found] Ollama is not running on localhost:11434.\n"
            "Please start the Ollama server and try again.")

    check_model = model if ":" in model else f"{model}:latest"
    try:
        tags_resp = requests.get("http://localhost:11434/api/tags",
                                 timeout=2).json()
        installed = [m.get("name") for m in tags_resp.get("models", [])]
        if check_model not in installed:
            sys.exit(
                f"[Model Not Found] The model '{model}' is not installed in Ollama.\n"
                f"Please open a terminal and run:\n  ollama pull {model}")
    except Exception as e:
        sys.exit(f"[Ollama Error] Failed to verify installed models: {e}")


if __name__ == "__main__":
    target_model = sys.argv[1] if len(sys.argv) > 1 else "qwen3.8:27b"
    run_preflight_checks(target_model)

    app = OllamaVisionApp(target_model)
    try:
        app.run()
    finally:
        app.cleanup()
