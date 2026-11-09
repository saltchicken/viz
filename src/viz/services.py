import json
from collections.abc import AsyncIterator

import httpx

from viz.config import Config


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
