import sys
from dataclasses import dataclass, field
from pathlib import Path

# Use the native tomllib in Python 3.11+, fallback to tomli in 3.10
if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib


@dataclass
class AppSettings:
    default_model: str = "qwen3.8:27b"
    required_commands: list[str] = field(default_factory=lambda: ["grim", "slurp", "wl-paste"])


@dataclass
class OllamaSettings:
    base_url: str = "http://localhost:11434"
    chat_timeout: int = 180
    status_timeout: int = 2


@dataclass
class PreviewSettings:
    max_width: int = 400
    max_height: int = 400
    margin: int = 8
    bg_color: str = "#1e1e2e"


@dataclass
class Config:
    app: AppSettings = field(default_factory=AppSettings)
    ollama: OllamaSettings = field(default_factory=OllamaSettings)
    preview: PreviewSettings = field(default_factory=PreviewSettings)
    presets: list[tuple[str, str]] = field(default_factory=lambda: [
        ("Custom", "Custom")
    ])

    @classmethod
    def load(cls, path: str = "config.toml") -> "Config":
        config_path = Path(path)
        if not config_path.exists():
            return cls()

        with open(config_path, "rb") as f:
            data = tomllib.load(f)

        app = AppSettings(**data.get("app", {}))
        ollama = OllamaSettings(**data.get("ollama", {}))
        preview = PreviewSettings(**data.get("preview", {}))

        presets_data = data.get("presets", [])
        presets = [(p.get("name", "Custom"), p.get("prompt", "Custom")) for p in presets_data]
        if not presets:
            presets = cls().presets

        return cls(app=app, ollama=ollama, preview=preview, presets=presets)


# Global configuration instance
config = Config.load()
