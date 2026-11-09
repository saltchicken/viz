import base64
import math
import subprocess
import tempfile
from datetime import datetime
from pathlib import Path

from PIL import Image

from viz.config import Config


class ImageWorkspace:
    """Manages temporary image storage, conversions, and external capture tools."""

    def __init__(self, config: Config) -> None:
        self.config = config
        # TemporaryDirectory automatically cleans up when garbage collected or closed
        self._temp_dir_obj = tempfile.TemporaryDirectory(
            prefix="ollama-vision-")
        self.tmp_dir = Path(self._temp_dir_obj.name)
        self.img_dir = self.tmp_dir / "images"
        self.img_dir.mkdir(parents=True, exist_ok=True)
        self.preview_path = self.tmp_dir / "preview.png"

    def get_attached_images(self) -> list[Path]:
        return sorted(self.img_dir.glob("*.png"),
                      key=lambda p: p.stat().st_mtime)

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

        images = [Image.open(p) for p in png_files]
        max_w = self.config.preview.max_width
        max_h = self.config.preview.max_height
        margin = self.config.preview.margin

        # Resize images, keeping aspect ratio
        processed_images = []
        for img in images:
            img.thumbnail((max_w, max_h))
            processed_images.append(img)

        # Calculate grid size (cols x rows)
        n = len(processed_images)
        cols = math.ceil(math.sqrt(n))
        rows = math.ceil(n / cols)

        # Calculate cell constraints and total canvas size
        cell_w = max(img.width for img in processed_images)
        cell_h = max(img.height for img in processed_images)

        canvas_w = cols * cell_w + (cols + 1) * margin
        canvas_h = rows * cell_h + (rows + 1) * margin

        canvas = Image.new("RGB", (canvas_w, canvas_h),
                           color=self.config.preview.bg_color)

        for idx, img in enumerate(processed_images):
            row = idx // cols
            col = idx % cols

            x = margin + col * (cell_w + margin)
            y = margin + row * (cell_h + margin)

            # Center the image within its designated grid cell
            offset_x = x + (cell_w - img.width) // 2
            offset_y = y + (cell_h - img.height) // 2

            # Use the image as its own mask if it has transparency
            if img.mode in ("RGBA", "LA") or (img.mode == "P" and
                                              "transparency" in img.info):
                canvas.paste(img, (offset_x, offset_y), img.convert("RGBA"))
            else:
                canvas.paste(img, (offset_x, offset_y))

        canvas.save(self.preview_path)
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
        self._temp_dir_obj.cleanup()

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
        try:
            with Image.open(src) as img:
                # Convert to RGBA if necessary to preserve transparency
                if img.mode in ("RGBA", "P"):
                    img = img.convert("RGBA")
                elif img.mode != "RGB":
                    img = img.convert("RGB")

                img.save(dst, format="PNG")
            return True
        except Exception:
            return False
