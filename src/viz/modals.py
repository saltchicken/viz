from pathlib import Path

from textual import on
from textual.app import ComposeResult
from textual.containers import Horizontal
from textual.containers import Vertical
from textual.screen import ModalScreen
from textual.widgets import Button
from textual.widgets import DirectoryTree
from textual.widgets import Label
from textual.widgets import Static


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
