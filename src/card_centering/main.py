"""Entry point for the card centering detector application.

Handles both development (python -m card_centering) and frozen (PyInstaller) modes.
"""

import sys
import os
import logging


def _setup_environment():
    """Configure runtime environment for both dev and frozen modes."""
    if getattr(sys, "frozen", False):
        # Running as PyInstaller bundle — the app root is sys._MEIPASS
        bundle_dir = sys._MEIPASS
        os.environ["CARD_CENTERING_ROOT"] = bundle_dir
        # Ensure Qt plugins path is set correctly
        qt_plugin_path = os.path.join(bundle_dir, "PySide6", "plugins")
        if os.path.isdir(qt_plugin_path):
            os.environ["QT_PLUGIN_PATH"] = qt_plugin_path
    else:
        # Running from source
        os.environ["CARD_CENTERING_ROOT"] = os.path.dirname(
            os.path.dirname(os.path.abspath(__file__))
        )

    from card_centering.platform_utils import get_log_path

    log_file = get_log_path()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8", delay=True),
            logging.StreamHandler(sys.stderr),
        ],
    )


def main():
    """Launch the PySide6 GUI application."""
    _setup_environment()

    from card_centering.gui import main as gui_main
    gui_main()


if __name__ == "__main__":
    main()
