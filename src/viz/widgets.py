import json
import re
import time

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import Button, Label, Markdown, Static


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
    .cmd-buttons {
        height: auto;
        align: right middle;
    }
    .cmd-buttons Button {
        margin-left: 1;
    }
    """

    def __init__(self, tool_call: dict, **kwargs) -> None:
        super().__init__(**kwargs)
        self.tool_call = tool_call
        
        args = tool_call.get("function", {}).get("arguments", {})
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except Exception:
                args = {}
        self.command = args.get("command", "")

    def compose(self) -> ComposeResult:
        yield Label("⚠️ The AI wants to execute a command:", classes="cmd-title")
        yield Static(self.command, classes="cmd-code")
        with Horizontal(classes="cmd-buttons"):
            yield Button("Decline", id="btn-decline-cmd", variant="error")
            yield Button("Accept & Run", id="btn-accept-cmd", variant="success")

    def mark_resolved(self) -> None:
        """Marks the command approval as resolved and disables interaction buttons."""
        self.add_class("resolved")
        for btn in self.query(Button):
            btn.disabled = True

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
        prefix_md, prefix_raw = self._get_role_prefixes()

        with Vertical(classes="md-view-container"):
            if self.streaming:
                self._streaming_md = Markdown(f"{prefix_md}{self.raw_text}",
                                              classes="md-chunk")
                yield self._streaming_md
            else:
                yield from self._compose_markdown_chunks(prefix_md)

        self.raw_view = Static(f"{prefix_raw}{self.raw_text}", markup=False)
        yield self.raw_view

        if self.streaming and not self.raw_text:
            self._status_label = Static(
                "⏳ Waiting for model... (Press Esc to cancel)",
                classes="stream-status",
            )
            yield self._status_label

    def _get_role_prefixes(self) -> tuple[str, str]:
        if self.role == "user":
            return "**You:**\n\n", "You:\n\n"
        if self.role in ("system", "tool"):
            return "**System:**\n\n", "System:\n\n"
        return (
            f"**Ollama ({self.model_name}):**\n\n",
            f"Ollama ({self.model_name}):\n\n",
        )

    def _compose_markdown_chunks(self, prefix_md: str) -> ComposeResult:
        # Match ``` at the start of a line (with up to 3 spaces of indentation)
        chunks = re.split(r"(^[ \t]{0,3}```[\s\S]*?(?:^[ \t]{0,3}```|\Z))", self.raw_text, flags=re.MULTILINE)
        if chunks:
            chunks[0] = prefix_md + chunks[0]

        for chunk in chunks:
            if not chunk.strip():
                continue

            if self._is_code_block(chunk):
                md_text = chunk
                # Ensure unclosed code blocks format correctly in the Markdown widget
                if not md_text.rstrip().endswith("```"):
                    md_text += "\n```"
                yield Markdown(md_text, classes="md-chunk")
            else:
                yield Markdown(chunk, classes="md-chunk")

    def append_chunk(self, chunk: str) -> None:
        """Appends a streamed token chunk and safely throttles markdown updates."""
        if self._status_label is not None:
            self._status_label.remove()
            self._status_label = None

        self.raw_text += chunk
        prefix_md, prefix_raw = self._get_role_prefixes()

        # Update the raw text view immediately (very cheap operation)
        self.raw_view.update(f"{prefix_raw}{self.raw_text}")
        
        # Throttle updates to the Markdown view to avoid quadratic CPU blocking
        if self._streaming_md is not None:
            now = time.monotonic()
            if now - self._last_md_update > self._update_throttle:
                self._streaming_md.update(f"{prefix_md}{self.raw_text}")
                self._last_md_update = now

    async def finalize_stream(self) -> None:
        """Finalizes streaming and forces a final render of code blocks."""
        self.streaming = False
        if self._status_label is not None:
            self._status_label.remove()
            self._status_label = None

        prefix_md, _ = self._get_role_prefixes()

        if "```" not in self.raw_text:
            if self._streaming_md is not None:
                self._streaming_md.update(f"{prefix_md}{self.raw_text}")
            return

        md_container = self.query_one(".md-view-container", Vertical)
        
        # DOM Operations must be awaited
        await md_container.remove_children()
        self._streaming_md = None

        chunks = re.split(r"(^[ \t]{0,3}```[\s\S]*?(?:^[ \t]{0,3}```|\Z))", self.raw_text, flags=re.MULTILINE)
        if chunks:
            chunks[0] = prefix_md + chunks[0]

        widgets_to_mount = []
        for chunk in chunks:
            if not chunk.strip():
                continue
            if self._is_code_block(chunk):
                md_text = chunk
                if not md_text.rstrip().endswith("```"):
                    md_text += "\n```"
                widgets_to_mount.append(Markdown(md_text, classes="md-chunk"))
            else:
                widgets_to_mount.append(Markdown(chunk, classes="md-chunk"))
                
        # Safely mount all assembled widgets in a single pass
        await md_container.mount_all(widgets_to_mount)

    @staticmethod
    def _is_code_block(chunk: str) -> bool:
        return chunk.lstrip().startswith("```")

    def on_mount(self) -> None:
        self.toggle_view(self.is_markdown)

    def toggle_view(self, show_markdown: bool) -> None:
        self.query_one(".md-view-container", Vertical).display = show_markdown
        self.raw_view.display = not show_markdown
