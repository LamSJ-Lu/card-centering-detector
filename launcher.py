"""
Standalone launcher for the TCG Card Centering Detector.

This script serves as the entry point for both:
    python launcher.py     (source mode)
    launcher.exe           (PyInstaller frozen mode)

It detects the execution context and runs the application accordingly.
"""

import sys
import os

# Add the source package to path when running from source
if not getattr(sys, "frozen", False):
    _src_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src")
    if _src_dir not in sys.path:
        sys.path.insert(0, _src_dir)


def _setup_qt_path():
    """Ensure Qt can find its plugins in frozen mode."""
    if getattr(sys, "frozen", False):
        bundle = sys._MEIPASS
        # PySide6 plugins directory
        for candidate in [
            os.path.join(bundle, "PySide6", "plugins"),
            os.path.join(bundle, "_internal", "PySide6", "plugins"),
        ]:
            if os.path.isdir(os.path.join(candidate, "platforms")):
                os.environ["QT_PLUGIN_PATH"] = candidate
                break


def main():
    _setup_qt_path()
    from card_centering.main import main as app_main
    app_main()


if __name__ == "__main__":
    main()
