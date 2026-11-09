from abc import ABC
from abc import abstractmethod
import asyncio
import base64
from collections.abc import AsyncIterator
from datetime import datetime
import json
import math
from pathlib import Path
import shlex
import subprocess
import tempfile

import httpx
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


def _build_ollama_payload(model: str,
                          messages: list[dict],
                          stream: bool,
                          options: dict | None = None,
                          tools: list[dict] | None = None) -> dict:
    """DRY helper for building API payload formatting constraints."""
    payload = {
        "model": model,
        "messages": messages,
        "stream": stream,
    }
    if options:
        payload["options"] = options
    if tools:
        payload["tools"] = tools
    return payload


async def query_ollama_chat_async(
    model: str,
    messages: list[dict],
    config: Config,
    options: dict | None = None,
) -> tuple[str, str | None]:
    """Sends a chat request to Ollama and returns (content, error_message)."""
    payload = _build_ollama_payload(model,
                                    messages,
                                    stream=False,
                                    options=options)

    try:
        async with httpx.AsyncClient(
                timeout=config.ollama.chat_timeout) as client:
            resp = await client.post(f"{config.ollama.base_url}/api/chat",
                                     json=payload)
            resp.raise_for_status()
            data = resp.json()
            error_msg = data.get("error")
            content = data.get("message", {}).get("content", "")
            return content, error_msg
    except Exception as e:
        return "", f"No response or connection timed out: {e}"


async def stream_ollama_chat_async(
    model: str,
    messages: list[dict],
    config: Config,
    options: dict | None = None,
    tools: list[dict] | None = None,
) -> AsyncIterator[tuple[str, str | None, dict | None, list[dict] | None]]:
    """Streams a chat request from Ollama natively, yielding (token_chunk, error_message, metrics, tool_calls)."""
    payload = _build_ollama_payload(model,
                                    messages,
                                    stream=True,
                                    options=options,
                                    tools=tools)

    try:
        async with httpx.AsyncClient(
                timeout=config.ollama.chat_timeout) as client:
            async with client.stream(
                    "POST",
                    f"{config.ollama.base_url}/api/chat",
                    json=payload,
            ) as resp:
                if resp.status_code != 200:
                    await resp.aread()
                    try:
                        err = resp.json().get("error", resp.text)
                    except Exception:
                        err = f"HTTP {resp.status_code}: {resp.text}"
                    yield "", err, None, None
                    return

                async for line in resp.aiter_lines():
                    if not line:
                        continue
                    data = json.loads(line)
                    if error_msg := data.get("error"):
                        yield "", error_msg, None, None
                        return

                    chunk = data.get("message", {}).get("content", "")
                    tool_calls = data.get("message", {}).get("tool_calls")
                    is_done = data.get("done", False)

                    # Ollama only sends eval counts in the final chunk when done=True
                    metrics = data if is_done else None

                    if chunk or metrics or tool_calls:
                        yield chunk, None, metrics, tool_calls

                    if is_done:
                        break
    except Exception as e:
        yield "", f"No response or connection timed out: {e}", None, None


# --- Tool Registry Implementation ---

def validate_command(
        command: str,
        allowed_commands: list[str]) -> tuple[bool, str, list[str]]:
    """Parses the command for vulnerabilities, validates args, and checks against allowlist."""
    if not command.strip():
        return False, "Command is empty.", []

    # 1. Parse command to securely tokenize arguments
    try:
        parsed_cmd = shlex.split(command)
    except ValueError as e:
        return False, f"Malformed command string: {e}", []

    if not parsed_cmd:
        return False, "Command resolved to empty after parsing.", []

    base_cmd = parsed_cmd[0]

    # 2. Check against the configured allowlist
    if base_cmd not in allowed_commands:
        return False, f"Command '{base_cmd}' is not in the allowed list.", []

    # 3. Check for shell injection vulnerabilities
    forbidden_chars = [";", "&", "|", "`", "$", "<", ">"]
    for char in forbidden_chars:
        if char in command:
            return False, f"Command contains forbidden shell operator: '{char}'. Chaining and piping are disabled.", []

    # 4. Restrict file paths to prevent directory traversal
    for arg in parsed_cmd[1:]:
        if arg.startswith("/") or ".." in arg:
            return False, f"Path traversal blocked: '{arg}'. Commands must operate in the current directory.", []

    return True, "", parsed_cmd


def resolve_secure_path(file_path: str) -> Path:
    """Resolves a file path and ensures it stays within the current working directory."""
    base_dir = Path.cwd().resolve()
    p = Path(file_path).resolve()

    if not p.is_relative_to(base_dir):
        raise PermissionError(
            "Security Policy Violation: Path traversal blocked. Path must be within the current working directory."
        )
    
    return p


class BaseTool(ABC):
    name: str

    @abstractmethod
    def get_schema(self) -> dict:
        """Returns the JSON schema payload for Ollama."""
        pass

    @abstractmethod
    def format_approval(self, args: dict) -> str:
        """Returns the formatted string to display in the approval UI."""
        pass

    @abstractmethod
    async def execute(self, args: dict) -> tuple[str, str]:
        """
        Executes the tool.
        Returns (internal_llm_output, user_facing_display_message).
        """
        pass


class ExecuteCommandTool(BaseTool):
    name = "execute_command"

    def __init__(self, config: Config):
        self.config = config

    def get_schema(self) -> dict:
        allowed_list = ", ".join(self.config.app.allowed_commands)

        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": (
                    "Executes a shell command on the user's system. "
                    f"STRICT ALLOWLIST - You are ONLY permitted to use these base commands: {allowed_list}. "
                    "SECURITY RESTRICTION: Do not use shell operators (|, &, ;, <, >, $, `). "
                    "Chaining and redirection are completely disabled and will cause the command to fail."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "command": {
                            "type": "string",
                            "description": "The exact command to execute."
                        }
                    },
                    "required": ["command"]
                }
            }
        }

    def format_approval(self, args: dict) -> str:
        return args.get("command", "")

    async def execute(self, args: dict) -> tuple[str, str]:
        command = args.get("command", "")
        is_secure, error_msg, parsed_cmd = validate_command(command, self.config.app.allowed_commands)
        
        if not is_secure:
            return f"Security Policy Violation: {error_msg}", f"**Command Blocked (Security):**\n```\n{error_msg}\n```"

        try:
            proc = await asyncio.create_subprocess_exec(
                *parsed_cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await proc.communicate()
            output = stdout.decode()
            if stderr:
                output += f"\n[STDERR]\n{stderr.decode()}"
        except Exception as e:
            output = f"Error executing command: {str(e)}"

        if not output.strip():
            output = "Command executed successfully with no output."

        return output, f"**Command Result:**\n```\n{output}\n```"


class ContextTool(BaseTool):
    name = "context"

    def get_schema(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": (
                    "Retrieves project directory structures and file contents. "
                    "Use this to explore the codebase and read file contents before answering questions."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "target_path": {
                            "type": "string",
                            "description": "Optional relative path to a specific file or directory (defaults to current directory)."
                        }
                    },
                    "required": []
                }
            }
        }

    def format_approval(self, args: dict) -> str:
        target = args.get("target_path", "").strip()
        return f"context {target}".strip()

    async def execute(self, args: dict) -> tuple[str, str]:
        target = args.get("target_path", "").strip()

        if target.startswith("/") or ".." in target:
            error_msg = f"Path traversal blocked: '{target}'. Must operate in the current directory."
            return f"Security Policy Violation: {error_msg}", f"**Context Blocked (Security):**\n```\n{error_msg}\n```"

        cmd = ["context", target] if target else ["context"]

        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await proc.communicate()
            output = stdout.decode()
            if stderr:
                output += f"\n[STDERR]\n{stderr.decode()}"
        except Exception as e:
            output = f"Error executing context: {str(e)}"

        if not output.strip():
            output = "Context executed successfully with no output."

        return output, f"**Context Result:**\n```\n{output}\n```"


class EditFileTool(BaseTool):
    name = "edit_file"

    def get_schema(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": "Edit a file by replacing an exact search string with a new string.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "file_path": {
                            "type": "string",
                            "description": "Path to the file to edit."
                        },
                        "search_string": {
                            "type": "string",
                            "description": "The exact block of text to search for. Must match the file contents perfectly including whitespace and indentation."
                        },
                        "replace_string": {
                            "type": "string",
                            "description": "The new block of text to replace it with."
                        }
                    },
                    "required": ["file_path", "search_string", "replace_string"]
                }
            }
        }

    def format_approval(self, args: dict) -> str:
        return (
            f"File: {args.get('file_path')}\n"
            f"--- SEARCH ---\n{args.get('search_string')}\n"
            f"--- REPLACE ---\n{args.get('replace_string')}"
        )

    async def execute(self, args: dict) -> tuple[str, str]:
        file_path = args.get("file_path", "")
        search_str = args.get("search_string", "")
        replace_str = args.get("replace_string", "")

        try:
            p = resolve_secure_path(file_path)

            if not p.exists():
                raise FileNotFoundError("File does not exist.")

            content = p.read_text(encoding="utf-8")
            if search_str not in content:
                raise ValueError("Exact search string not found in the file.")

            # Replace only the first occurrence to avoid accidental overwrites
            new_content = content.replace(search_str, replace_str, 1)
            p.write_text(new_content, encoding="utf-8")

            return f"Successfully edited {file_path}.", f"**File Edited Successfully:** `{file_path}`"
        except Exception as e:
            return f"Error editing file: {str(e)}", f"**Edit Failed:**\n```\n{e}\n```"


class CreateFileTool(BaseTool):
    name = "create_file"

    def get_schema(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": "Create a new file with the specified text content.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "file_path": {
                            "type": "string",
                            "description": "Path where the file should be created."
                        },
                        "content": {
                            "type": "string",
                            "description": "The text content to write into the new file."
                        }
                    },
                    "required": ["file_path", "content"]
                }
            }
        }

    def format_approval(self, args: dict) -> str:
        return (
            f"Create File: {args.get('file_path')}\n"
            f"--- CONTENT ---\n{args.get('content')}"
        )

    async def execute(self, args: dict) -> tuple[str, str]:
        file_path = args.get("file_path", "")
        content = args.get("content", "")

        try:
            p = resolve_secure_path(file_path)
            
            if p.exists():
                raise FileExistsError("File already exists. Use edit_file instead.")
                
            # Automatically create parent directories if they don't exist
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(content, encoding="utf-8")

            return f"Successfully created {file_path}.", f"**File Created Successfully:** `{file_path}`"
        except Exception as e:
            return f"Error creating file: {str(e)}", f"**Creation Failed:**\n```\n{e}\n```"


class ToolRegistry:
    def __init__(self, config: Config):
        self.tools: dict[str, BaseTool] = {}
        # Register core tools
        self.register(ExecuteCommandTool(config))
        self.register(ContextTool())
        self.register(EditFileTool())
        self.register(CreateFileTool())

    def register(self, tool: BaseTool) -> None:
        self.tools[tool.name] = tool

    def get_schemas(self) -> list[dict]:
        return [tool.get_schema() for tool in self.tools.values()]

    def get_tool(self, name: str) -> BaseTool | None:
        return self.tools.get(name)
