import base64
from datetime import datetime
from pathlib import Path
import shutil
import subprocess
import tempfile
import requests

OLLAMA_BASE_URL = "http://localhost:11434"


class ImageWorkspace:
    """Manages temporary image storage, conversions, and external capture tools."""

    def __init__(self) -> None:
        self.tmp_dir = Path(tempfile.mkdtemp(prefix="ollama-vision-"))
        self.img_dir = self.tmp_dir / "images"
        self.img_dir.mkdir(parents=True, exist_ok=True)
        self.preview_path = self.tmp_dir / "preview.png"

    def get_attached_images(self) -> list[Path]:
        return sorted(self.img_dir.glob("*.png"), key=lambda p: p.stat().st_mtime)

    def get_encoded_images(self) -> list[str]:
        return [
            base64.b64encode(p.read_bytes()).decode("utf-8")
            for p in self.get_attached_images()
        ]

    def build_preview(self) -> Path | None:
        png_files = self.get_attached_images()
        if not png_files:
            return None
        if len(png_files) == 1:
            return png_files[0]

        cmd = ["magick", "montage"] if shutil.which("magick") else ["montage"]
        subprocess.run(
            [
                *cmd,
                "-background",
                "#1e1e2e",
                "-geometry",
                "400x400>+8+8",
                *[str(p) for p in png_files],
                str(self.preview_path),
            ],
            capture_output=True,
        )
        return self.preview_path

    def capture_screen_region(self) -> bool:
        img_file = self._generate_filepath("shot")
        slurp = subprocess.run(["slurp"], capture_output=True, text=True)
        if slurp.returncode == 0 and slurp.stdout.strip():
            geom = slurp.stdout.strip()
            subprocess.run(["grim", "-g", geom, str(img_file)])

        return self._validate_file(img_file)

    def paste_clipboard_image(self) -> bool:
        img_file = self._generate_filepath("paste")
        with open(img_file, "wb") as f:
            subprocess.run(
                ["wl-paste", "-t", "image/png"],
                stdout=f,
                stderr=subprocess.DEVNULL,
            )
        return self._validate_file(img_file)

    def import_files(self, files: list[Path]) -> int:
        added = 0
        for f in files:
            if f.is_file():
                out = self._generate_filepath("file", suffix=f"_{added}")
                if self._convert_image(f, out):
                    added += 1
        return added

    def clear_images(self) -> None:
        for p in self.img_dir.glob("*.png"):
            p.unlink(missing_ok=True)

    def cleanup(self) -> None:
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def _generate_filepath(self, prefix: str, suffix: str = "") -> Path:
        timestamp = datetime.now().strftime("%s%f")
        return self.img_dir / f"{prefix}_{timestamp}{suffix}.png"

    @staticmethod
    def _validate_file(path: Path) -> bool:
        if path.exists() and path.stat().st_size > 0:
            return True
        path.unlink(missing_ok=True)
        return False

    @staticmethod
    def _convert_image(src: Path, dst: Path) -> bool:
        cmd = "magick" if shutil.which("magick") else "convert"
        res = subprocess.run([cmd, str(src), str(dst)], capture_output=True)
        return res.returncode == 0


def query_ollama_chat(model: str, messages: list[dict], timeout: int = 180) -> tuple[str, str | None]:
    """Sends a chat request to Ollama and returns (content, error_message)."""
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
    }
    try:
        resp = requests.post(f"{OLLAMA_BASE_URL}/api/chat", json=payload, timeout=timeout)
        data = resp.json()
        error_msg = data.get("error")
        content = data.get("message", {}).get("content", "")
        return content, error_msg
    except Exception as e:
        return "", f"No response or connection timed out: {e}"
