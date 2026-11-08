#!/usr/bin/env python3
import sys
import os
import shutil
import tempfile
import subprocess
import base64
from pathlib import Path
from datetime import datetime
import requests

from textual import on, work
from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.widgets import (
    Header, Footer, Button, Select, TextArea, Static, 
    Markdown, Label, DirectoryTree
)
from textual.screen import ModalScreen
from textual.worker import get_current_worker

# Use textual-image for native Kitty/Sixel terminal pixel rendering
from textual_image.widget import Image as TextualImage


PRESETS = [
    ("Custom", "Custom"),
    (
        "Describe this image in detail.", 
        "Describe this image in detail."
    ),
    (
        "Extract all the text from this image.", 
        "Extract all the text from this image."
    ),
    (
        "Create FLUX3 Prompt", 
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
            yield Label("[b]Select Image(s)[/b] (Click files to toggle selection)")
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
        self.query_one("#selected-label", Static).update(
            f"Selected ({len(self.selected_files)}): {names}"
        )

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
                id="thinking-text"
            )
            yield Button("Cancel Request", id="cancel-req-btn", variant="error")

    @on(Button.Pressed, "#cancel-req-btn")
    def cancel_request(self) -> None:
        self.dismiss(False)


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
    #preview-state {
        height: auto;
        color: $text-muted;
        margin-bottom: 1;
    }
    #preview-image {
        height: 14;
        width: 100%;
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
    .user-msg {
        background: $boost;
        border-left: thick $primary;
        padding: 1;
        margin-bottom: 1;
    }
    .assistant-msg {
        background: $surface;
        border-left: thick $success;
        padding: 1;
        margin-bottom: 1;
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
                yield Select(PRESETS, value="Custom", id="preset-select", allow_blank=False)
                
                yield Label("Prompt (Multiline):", id="prompt-label")
                yield TextArea(id="prompt-input")
                
                with Horizontal(classes="btn-row"):
                    yield Button("Ask Ollama (Ctrl+S)", id="btn-ask", variant="success")
                    yield Button("Reset Chat", id="btn-reset", variant="error")

            # Right Panel: Markdown Chat History
            with Vertical(id="chat-panel"):
                yield Label("Conversation History", classes="section-title")
                with VerticalScroll(id="chat-scroll"):
                    yield Static(
                        "Attach images and enter a prompt on the left to start chatting.",
                        id="empty-chat-hint"
                    )
        yield Footer()

    def on_mount(self) -> None:
        self.update_preview()

    def update_preview(self) -> None:
        # Sort by modification time so images appear in the exact order they were added
        png_files = sorted(self.img_dir.glob("*.png"), key=lambda p: p.stat().st_mtime)
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
            # Show single image directly at full resolution
            img_widget.image = png_files[0]
        else:
            # Stitch multiple images into a high-res gallery preview
            cmd = ["magick", "montage"] if shutil.which("magick") else ["montage"]
            subprocess.run([
                *cmd,
                "-background", "#1e1e2e",
                "-geometry", "400x400>+8+8",
                *[str(p) for p in png_files],
                str(preview_path),
            ], capture_output=True)
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

        # Suspend the TUI temporarily so slurp/grim can grab the screen cleanly
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
            subprocess.run(
                ["wl-paste", "-t", "image/png"],
                stdout=f,
                stderr=subprocess.DEVNULL
            )

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
                self.notify("Failed to convert selected file(s).", severity="error")
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
            Static("Chat reset. Attach images and ask a question.", id="empty-chat-hint")
        )
        self.query_one("#prompt-label", Label).update("Prompt (Multiline):")
        self.notify("Conversation history cleared.")

    @on(Button.Pressed, "#btn-ask")
    def action_ask_ollama(self) -> None:
        preset_val = self.query_one("#preset-select", Select).value
        custom_text = self.query_one("#prompt-input", TextArea).text.strip()

        prompt = custom_text if preset_val == "Custom" else (custom_text or str(preset_val))
        if not prompt:
            self.notify("Prompt cannot be empty.", severity="error")
            return

        # Remove initial hint if present
        for hint in self.query("#empty-chat-hint"):
            hint.remove()

        # Build message payload
        msg_entry: dict = {"role": "user", "content": prompt}
        if self.first_turn:
            # Sort by modification time so they are sent to Ollama in the correct order
            png_files = sorted(self.img_dir.glob("*.png"), key=lambda p: p.stat().st_mtime)
            if png_files:
                b64_images = [
                    base64.b64encode(p.read_bytes()).decode("utf-8")
                    for p in png_files
                ]
                msg_entry["images"] = b64_images
            self.first_turn = False
            self.query_one("#prompt-label", Label).update("Follow-up Question:")

        self.messages.append(msg_entry)

        # Mount User Message in Chat UI
        chat_scroll = self.query_one("#chat-scroll", VerticalScroll)
        user_md = Markdown(f"**You:**\n\n{prompt}", classes="user-msg")
        chat_scroll.mount(user_md)
        chat_scroll.scroll_end(animate=False)

        # Clear input for next turn
        self.query_one("#prompt-input", TextArea).clear()
        self.query_one("#preset-select", Select).value = "Custom"

        # Show Thinking Modal & start background API worker
        self.thinking_modal = ThinkingModal(prompt)
        
        def on_modal_close(completed_normally: bool | None) -> None:
            if completed_normally is False and self.active_worker:
                self.active_worker.cancel()
                # Roll back the unanswered user message from history
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
            resp = requests.post(
                "http://localhost:11434/api/chat",
                json=payload,
                timeout=180
            )
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
            self.notify("Received empty response from the model.", severity="error")
            return

        # Save to history
        self.messages.append({"role": "assistant", "content": content})

        # Copy response to Wayland clipboard automatically
        subprocess.run(["wl-copy"], input=content.encode("utf-8"), stderr=subprocess.DEVNULL)

        # Render Markdown in Chat UI
        chat_scroll = self.query_one("#chat-scroll", VerticalScroll)
        assistant_md = Markdown(f"**Ollama ({self.model}):**\n\n{content}", classes="assistant-msg")
        chat_scroll.mount(assistant_md)
        chat_scroll.scroll_end(animate=True)
        self.notify("Response received and copied to clipboard.")

    def cleanup(self) -> None:
        shutil.rmtree(self.tmp_dir, ignore_errors=True)


def run_preflight_checks(model: str) -> None:
    """Replicates steps 2, 3, and 3b of viz.sh before launching the TUI."""
    # 2. Dependency Checks
    required_cmds = ["grim", "slurp", "wl-copy", "wl-paste"]
    for cmd in required_cmds:
        if not shutil.which(cmd):
            sys.exit(f"[Missing Dependency] Required command '{cmd}' is not installed.")

    if not shutil.which("magick") and not shutil.which("convert"):
        sys.exit("[Missing Dependency] ImageMagick is required. Please install 'imagemagick'.")

    # 3. Check if Ollama is running
    try:
        r = requests.get("http://localhost:11434", timeout=2)
        r.raise_for_status()
    except Exception:
        sys.exit(
            "[Ollama Not Found] Ollama is not running on localhost:11434.\n"
            "Please start the Ollama server and try again."
        )

    # 3b. Check if the specified model is downloaded
    check_model = model if ":" in model else f"{model}:latest"
    try:
        tags_resp = requests.get("http://localhost:11434/api/tags", timeout=2).json()
        installed = [m.get("name") for m in tags_resp.get("models", [])]
        if check_model not in installed:
            sys.exit(
                f"[Model Not Found] The model '{model}' is not installed in Ollama.\n"
                f"Please open a terminal and run:\n  ollama pull {model}"
            )
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
