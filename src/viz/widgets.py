import json
import re
import time

from textual.app import ComposeResult
from textual.containers import Horizontal
from textual.containers import Vertical
from textual.widgets import Button
from textual.widgets import Input
from textual.widgets import Label
from textual.widgets import Markdown
from textual.widgets import Static

from viz.services import ToolRegistry


class CommandApproval(Vertical):
    """Widget to ask user for command execution approval."""

    DEFAULT_CSS = """
    CommandApproval {
        background: $boost;
        border: solid $warning;
        padding: 1;
        margin-bottom: 1;
        height: auto;
    }
    .cmd-title {
        text-style: bold;
        color: $warning;
        margin-bottom: 1;
    }
    .cmd-code {
        background: $surface;
        padding: 1;
        margin-bottom: 1;
        height: auto;
    }
    #cmd-feedback-input {
        margin-bottom: 1;
    }
    .cmd-buttons {
        height: auto;
        align: right middle;
    }
    .cmd-buttons Button {
        margin-left: 1;
    }
    """

    def __init__(self, tool_call: dict, tool_registry: ToolRegistry, **kwargs) -> None:
        super().__init__(**kwargs)
        self.tool_call = tool_call
        self.tool_name = tool_call.get("function", {}).get("name", "unknown")
        self.tool_registry = tool_registry

        args = tool_call.get("function", {}).get("arguments", {})
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except Exception:
                args = {}
        self.args = args

    def compose(self) -> ComposeResult:
        yield Label(f"⚠️ The AI wants to use a tool: {self.tool_name}",
                    classes="cmd-title")

        # Dynamically fetch formatting from the tool itself
        tool = self.tool_registry.get_tool(self.tool_name)
        if tool:
            display_text = tool.format_approval(self.args)
        else:
            display_text = json.dumps(self.args, indent=2)

        yield Static(display_text, classes="cmd-code")
        yield Input(placeholder="Optional message to send with results...", id="cmd-feedback-input")
        
        with Horizontal(classes="cmd-buttons"):
            yield Button("Decline", id="btn-decline-cmd", variant="error")
            yield Button("Accept & Run", id="btn-accept-cmd", variant="success")

    def mark_resolved(self) -> None:
        """Marks the command approval as resolved and disables interaction buttons."""
        self.add_class("resolved")
        for btn in self.query(Button):
            btn.disabled = True
        for inp in self.query(Input):
            inp.disabled = True

    def on_mount(self) -> None:
        self.query_one("#btn-decline-cmd", Button).focus()


class ChatMessage(Vertical):
    """A container holding both Markdown and Raw text views, supporting live streaming."""

    DEFAULT_CSS = """
    .stream-status {
        color: $warning;
        text-style: italic;
        height: auto;
        margin-top: 1;
    }
    """

    def __init__(
        self,
        raw_text: str,
        role: str,
        model_name: str = "",
        is_markdown: bool = True,
        streaming: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.raw_text = raw_text
        self.role = role
        self.model_name = model_name
        self.is_markdown = is_markdown
        self.streaming = streaming
        self._streaming_md: Markdown | None = None
        self._status_label: Static | None = None

        # State to prevent O(N^2) Markdown widget parsing freeze during high-frequency streaming
        self._last_md_update: float = 0.0
        self._update_throttle: float = 0.1  # Update markdown at most every 100ms

    def compose(self) -> ComposeResult:
        with Vertical(classes="md-view-container"):
            if self.streaming:
                self._streaming_md = Markdown(self.raw_text, classes="md-chunk")
                yield self._streaming_md
            else:
                yield from self._build_markdown_widgets(self.raw_text)

        self.raw_view = Static(self.raw_text, markup=False)
        yield self.raw_view

        if self.streaming and not self.raw_text:
            self._status_label = Static(
                "⏳ Waiting for model... (Press Esc to cancel)",
                classes="stream-status",
            )
            yield self._status_label

    def _build_markdown_widgets(self, text: str) -> list[Markdown]:
        """DRY method for chunking text safely by block formatting and converting them to Markdown Widgets."""
        chunks = re.split(r"(^[ \t]{0,3}```[\s\S]*?(?:^[ \t]{0,3}```|\Z))",
                          text,
                          flags=re.MULTILINE)

        widgets = []
        for chunk in chunks:
            if not chunk.strip():
                continue

            if self._is_code_block(chunk):
                md_text = chunk
                # Ensure unclosed code blocks format correctly in the Markdown widget
                if not md_text.rstrip().endswith("```"):
                    md_text += "\n```"
                widgets.append(Markdown(md_text, classes="md-chunk"))
            else:
                widgets.append(Markdown(chunk, classes="md-chunk"))

        return widgets

    def append_chunk(self, chunk: str) -> None:
        """Appends a streamed token chunk and safely throttles markdown updates."""
        if self._status_label is not None:
            self._status_label.remove()
            self._status_label = None

        self.raw_text += chunk

        # Update the raw text view immediately (very cheap operation)
        self.raw_view.update(self.raw_text)

        # Throttle updates to the Markdown view to avoid quadratic CPU blocking
        if self._streaming_md is not None:
            now = time.monotonic()
            if now - self._last_md_update > self._update_throttle:
                self._streaming_md.update(self.raw_text)
                self._last_md_update = now

    async def finalize_stream(self) -> None:
        """Finalizes streaming and forces a final render of code blocks."""
        self.streaming = False
        if self._status_label is not None:
            self._status_label.remove()
            self._status_label = None

        if "```" not in self.raw_text:
            if self._streaming_md is not None:
                self._streaming_md.update(self.raw_text)
            return

        md_container = self.query_one(".md-view-container", Vertical)

        # DOM Operations must be awaited
        await md_container.remove_children()
        self._streaming_md = None

        # Safely mount all assembled widgets in a single pass
        widgets_to_mount = self._build_markdown_widgets(self.raw_text)
        await md_container.mount_all(widgets_to_mount)

    @staticmethod
    def _is_code_block(chunk: str) -> bool:
        return chunk.lstrip().startswith("```")

    def on_mount(self) -> None:
        self.toggle_view(self.is_markdown)

    def toggle_view(self, show_markdown: bool) -> None:
        self.query_one(".md-view-container", Vertical).display = show_markdown
        self.raw_view.display = not show_markdown
