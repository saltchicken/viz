import json
from collections.abc import AsyncIterator
from datetime import datetime
from pathlib import Path

import httpx

from viz.config import Config


class SessionManager:
    """Manages automatic saving and loading of text-only conversation sessions."""

    def __init__(self, config: Config) -> None:
        self.history_dir = Path(config.app.history_dir).expanduser().resolve()
        self.history_dir.mkdir(parents=True, exist_ok=True)

    def create_session_id(self) -> str:
        return datetime.now().strftime("%Y%m%d_%H%M%S_%f")

    def save_session(
        self,
        session_id: str,
        model: str,
        messages: list[dict],
        system_prompt: str = "",
        num_ctx: int = 32768,
    ) -> Path | None:
        if not messages:
            return None

        clean_messages = []
        for msg in messages:
            # Strip base64 images and internal private tracking fields (e.g., _source_paths)
            clean_msg = {
                k: v
                for k, v in msg.items()
                if not k.startswith("_") and k != "images"
            }
            # Preserve a lightweight note in the UI message dict if images were originally attached
            if "images" in msg:
                clean_msg["had_images"] = len(msg["images"])
            clean_messages.append(clean_msg)

        title = "Untitled Session"
        for m in clean_messages:
            if m.get("role") == "user" and m.get("content"):
                raw_title = m["content"].strip().splitlines()[0]
                title = (raw_title[:60] + "...") if len(raw_title) > 60 else raw_title
                break

        file_path = self.history_dir / f"session_{session_id}.json"
        created_at = datetime.now().isoformat(timespec="seconds")
        if file_path.exists():
            try:
                existing = json.loads(file_path.read_text(encoding="utf-8"))
                created_at = existing.get("created_at", created_at)
            except Exception:
                pass

        payload = {
            "id": session_id,
            "title": title,
            "model": model,
            "created_at": created_at,
            "updated_at": datetime.now().isoformat(timespec="seconds"),
            "system_prompt": system_prompt,
            "num_ctx": num_ctx,
            "messages": clean_messages,
        }

        file_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        return file_path

    def list_sessions(self) -> list[dict]:
        sessions = []
        for p in self.history_dir.glob("session_*.json"):
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
                data["_path"] = p
                sessions.append(data)
            except Exception:
                continue

        sessions.sort(key=lambda s: s.get("updated_at", ""), reverse=True)
        return sessions

    def load_session(self, path: Path) -> dict | None:
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return None

    def delete_session(self, path: Path) -> bool:
        try:
            path.unlink(missing_ok=True)
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
