import sys

from viz.app import OllamaVisionApp
from viz.config import Config
from viz.preflight import run_preflight_checks


def main() -> None:
    config = Config.load()
    target_model = sys.argv[1] if len(sys.argv) > 1 else config.app.default_model
    
    run_preflight_checks(target_model, config)

    app = OllamaVisionApp(target_model, config)
    try:
        app.run()
    finally:
        app.cleanup()


if __name__ == "__main__":
    main()
