import json
from pathlib import Path

from textual import on
from textual.app import ComposeResult
from textual.containers import Horizontal
from textual.containers import Vertical
from textual.screen import ModalScreen
from textual.widgets import Button
from textual.widgets import DirectoryTree
from textual.widgets import Label
from textual.widgets import OptionList
from textual.widgets import Static
from textual.widgets import TextArea
from textual.widgets.option_list import Option

from viz.services import SessionManager


class FileSelectModal(ModalScreen[list[Path]]):
    """Modal dialog to browse and select image files."""

    def __init__(self) -> None:
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
        self._toggle_file_selection(event.path)
        self._update_selection_label()

    def _toggle_file_selection(self, path: Path) -> None:
        if path in self.selected_files:
            self.selected_files.remove(path)
        else:
            self.selected_files.append(path)

    def _update_selection_label(self) -> None:
        names = ", ".join(p.name for p in self.selected_files)
        count = len(self.selected_files)
        self.query_one("#selected-label",
                       Static).update(f"Selected ({count}): {names}")

    @on(Button.Pressed, "#cancel-btn")
    def on_cancel(self) -> None:
        self.dismiss([])

    @on(Button.Pressed, "#add-btn")
    def on_add(self) -> None:
        self.dismiss(self.selected_files)


class DebugModal(ModalScreen[None]):
    """Modal dialog to show raw API request payload."""

    def __init__(self, payload: dict) -> None:
        super().__init__()
        self.payload = payload

    def compose(self) -> ComposeResult:
        with Vertical(id="debug-dialog"):
            yield Label("[b]Last Request Payload (Images Truncated)[/b]")
            yield TextArea(json.dumps(self.payload, indent=2),
                           language="json",
                           read_only=True,
                           id="debug-textarea")
            with Horizontal(classes="modal-buttons"):
                yield Button("Close", id="close-debug-btn", variant="primary")

    @on(Button.Pressed, "#close-debug-btn")
    def on_close(self) -> None:
        self.dismiss()


class SessionSelectModal(ModalScreen[dict | None]):
    """Modal dialog to browse, load, or delete saved conversation sessions."""

    def __init__(self, session_manager: SessionManager) -> None:
        super().__init__()
        self.session_manager = session_manager
        self.sessions: list[dict] = []

    def compose(self) -> ComposeResult:
        with Vertical(id="session-dialog"):
            yield Label(
                "[b]Saved Conversations[/b] (Enter/Double-click to Load)")
            yield OptionList(id="session-list")
            yield Static("", id="session-meta-label")
            with Horizontal(classes="modal-buttons"):
                yield Button("Cancel", id="cancel-session-btn")
                yield Button("Delete", id="delete-session-btn", variant="error")
                yield Button("Load", id="load-session-btn", variant="success")

    def on_mount(self) -> None:
        self._refresh_list()

    def _refresh_list(self) -> None:
        self.sessions = self.session_manager.list_sessions()
        opt_list = self.query_one("#session-list", OptionList)
        opt_list.clear_options()

        if not self.sessions:
            self.query_one("#session-meta-label", Static).update(
                f"No saved sessions found in {self.session_manager.history_dir}"
            )
            self.query_one("#load-session-btn", Button).disabled = True
            self.query_one("#delete-session-btn", Button).disabled = True
            return

        self.query_one("#load-session-btn", Button).disabled = False
        self.query_one("#delete-session-btn", Button).disabled = False

        for idx, s in enumerate(self.sessions):
            updated = s.get("updated_at", "").replace("T", " ")
            title = s.get("title", "Untitled")
            msg_count = len(s.get("messages", []))
            prompt = f"[{updated}] {title} ({msg_count} msgs)"
            opt_list.add_option(Option(prompt, id=str(idx)))

        opt_list.highlighted = 0
        self._update_meta(0)
        opt_list.focus()

    def _update_meta(self, idx: int) -> None:
        if 0 <= idx < len(self.sessions):
            s = self.sessions[idx]
            model = s.get("model", "unknown")
            path = s.get("_path", "")
            self.query_one("#session-meta-label", Static).update(
                f"Model: {model} | File: {path}"
            )

    @on(OptionList.OptionHighlighted, "#session-list")
    def on_highlighted(self, event: OptionList.OptionHighlighted) -> None:
        if event.option_index is not None:
            self._update_meta(event.option_index)

    @on(OptionList.OptionSelected, "#session-list")
    def on_option_selected(self, event: OptionList.OptionSelected) -> None:
        if event.option_index is not None and 0 <= event.option_index < len(self.sessions):
            self.dismiss(self.sessions[event.option_index])

    @on(Button.Pressed, "#load-session-btn")
    def on_load(self) -> None:
        opt_list = self.query_one("#session-list", OptionList)
        idx = opt_list.highlighted
        if idx is not None and 0 <= idx < len(self.sessions):
            self.dismiss(self.sessions[idx])

    @on(Button.Pressed, "#delete-session-btn")
    def on_delete(self) -> None:
        opt_list = self.query_one("#session-list", OptionList)
        idx = opt_list.highlighted
        if idx is not None and 0 <= idx < len(self.sessions):
            target = self.sessions[idx]
            path = target.get("_path")
            if path and self.session_manager.delete_session(path):
                self._refresh_list()

    @on(Button.Pressed, "#cancel-session-btn")
    def on_cancel(self) -> None:
        self.dismiss(None)
