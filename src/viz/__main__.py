import sys
from viz.app import OllamaVisionApp
from viz.preflight import run_preflight_checks

DEFAULT_MODEL = "qwen3.8:27b"


def main() -> None:
    target_model = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_MODEL
    run_preflight_checks(target_model)

    app = OllamaVisionApp(target_model)
    try:
        app.run()
    finally:
        app.cleanup()


if __name__ == "__main__":
    main()
