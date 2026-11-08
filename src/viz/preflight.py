import shutil
import sys

import httpx

from viz.config import config


def run_preflight_checks(model: str) -> None:
    """Verifies system dependencies, Ollama server status, and model availability."""
    _check_system_binaries()
    _check_ollama_server()
    _check_model_installed(model)


def _check_system_binaries() -> None:
    for cmd in config.app.required_commands:
        if not shutil.which(cmd):
            sys.exit(
                f"[Missing Dependency] Required command '{cmd}' is not installed."
            )


def _check_ollama_server() -> None:
    try:
        r = httpx.get(config.ollama.base_url, timeout=config.ollama.status_timeout)
        r.raise_for_status()
    except Exception:
        sys.exit(
            "[Ollama Not Found] Ollama is not running on localhost:11434.\n"
            "Please start the Ollama server and try again.")


def _check_model_installed(model: str) -> None:
    check_model = model if ":" in model else f"{model}:latest"
    try:
        tags_resp = httpx.get(f"{config.ollama.base_url}/api/tags",
                                 timeout=config.ollama.status_timeout).json()
        installed = [m.get("name") for m in tags_resp.get("models", [])]
        if check_model not in installed:
            sys.exit(
                f"[Model Not Found] The model '{model}' is not installed in Ollama.\n"
                f"Please open a terminal and run:\n  ollama pull {model}")
    except Exception as e:
        sys.exit(f"[Ollama Error] Failed to verify installed models: {e}")
