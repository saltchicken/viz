import re
import subprocess

from textual import on
from textual.app import ComposeResult
from textual.containers import Vertical
from textual.widgets import Button
from textual.widgets import Markdown
from textual.widgets import Static


class CopyCodeButton(Button):
    """A button that holds the specific code block text to be copied."""

    def __init__(self, code_text: str, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.code_text = code_text


class ChatMessage(Vertical):
    """A container holding both Markdown and Raw text views."""

    DEFAULT_CSS = """
    .code-wrapper {
        height: auto;
    }
    .code-wrapper .inline-copy-btn {
        dock: right;
        margin-top: 1;
        margin-right: 1;
    }
    """

    def __init__(
        self,
        raw_text: str,
        role: str,
        model_name: str = "",
        is_markdown: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.raw_text = raw_text
        self.role = role
        self.model_name = model_name
        self.is_markdown = is_markdown

    def compose(self) -> ComposeResult:
        prefix_md, prefix_raw = self._get_role_prefixes()

        with Vertical(classes="md-view-container"):
            yield from self._compose_markdown_chunks(prefix_md)

        self.raw_view = Static(f"{prefix_raw}{self.raw_text}", markup=False)
        yield self.raw_view

    def _get_role_prefixes(self) -> tuple[str, str]:
        if self.role == "user":
            return "**You:**\n\n", "You:\n\n"
        return f"**Ollama ({self.model_name}):**\n\n", f"Ollama ({self.model_name}):\n\n"

    def _compose_markdown_chunks(self, prefix_md: str) -> ComposeResult:
        chunks = re.split(r"(```.*?```)", self.raw_text, flags=re.DOTALL)
        if chunks:
            chunks[0] = prefix_md + chunks[0]

        for chunk in chunks:
            if not chunk.strip():
                continue

            if self._is_code_block(chunk):
                with Vertical(classes="code-wrapper"):
                    yield Markdown(chunk, classes="md-chunk")
                    yield CopyCodeButton(
                        self._extract_inner_code(chunk),
                        label="📋",
                        classes="copy-code-btn inline-copy-btn",
                        variant="primary",
                    )
            else:
                yield Markdown(chunk, classes="md-chunk")

    @staticmethod
    def _is_code_block(chunk: str) -> bool:
        return chunk.startswith("```") and chunk.endswith("```")

    @staticmethod
    def _extract_inner_code(chunk: str) -> str:
        match = re.match(r"```[^\n]*\n(.*?)```", chunk, re.DOTALL)
        return match.group(1).strip() if match else chunk.strip("`").strip()

    def on_mount(self) -> None:
        self.toggle_view(self.is_markdown)

    def toggle_view(self, show_markdown: bool) -> None:
        self.query_one(".md-view-container", Vertical).display = show_markdown
        self.raw_view.display = not show_markdown

    @on(Button.Pressed, ".copy-code-btn")
    def on_copy_code(self, event: Button.Pressed) -> None:
        button = event.button
        assert isinstance(button, CopyCodeButton)
        self._copy_text_to_clipboard(button.code_text, str(button.label))

    def _copy_text_to_clipboard(self, text: str, label: str) -> None:
        try:
            self.app.copy_to_clipboard(text)
            self.app.notify(f"Copied {label} to clipboard!")
        except Exception:
            try:
                subprocess.run(["wl-copy"],
                               input=text.encode("utf-8"),
                               check=True)
                self.app.notify(f"Copied {label} via wl-copy!")
            except Exception as e:
                self.app.notify(f"Clipboard error: {e}", severity="error")
