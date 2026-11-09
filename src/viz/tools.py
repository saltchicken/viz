import asyncio
import json
import shlex
from abc import ABC, abstractmethod
from pathlib import Path

from viz.config import Config


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
