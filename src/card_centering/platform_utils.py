"""Platform-specific utilities for cross-platform compatibility.

Provides a single source of truth for all platform-conditional logic:
camera backends, log paths, and Qt plugin path setup.
"""

import os
import sys

# ── Platform detection ────────────────────────────────────────────────────

IS_MACOS = sys.platform == "darwin"
IS_WINDOWS = sys.platform == "win32"
IS_LINUX = sys.platform.startswith("linux")


# ── Camera backend ────────────────────────────────────────────────────────

def get_camera_backend():
    """Return the appropriate OpenCV VideoCapture backend for the current OS.

    macOS      → AVFoundation (FaceTime / USB cameras)
    Windows    → DirectShow (better device enumeration than MSMF)
    Linux/other → AUTO (V4L2 detection)

    Uses ``getattr`` to safely fall back to ``CAP_ANY`` on builds where the
    platform-specific constant is missing.
    """
    import cv2

    if IS_MACOS:
        return getattr(cv2, "CAP_AVFOUNDATION", cv2.CAP_ANY)
    elif IS_WINDOWS:
        return getattr(cv2, "CAP_DSHOW", cv2.CAP_ANY)
    else:
        return cv2.CAP_ANY


# ── Log path ──────────────────────────────────────────────────────────────

def get_log_path(filename: str = "card_centering.log") -> str:
    """Return a platform-appropriate log file path.

    * macOS   → ``~/Library/Logs/<filename>``
    * Windows → ``%APPDATA%/<filename>``
    * Linux   → ``$XDG_STATE_HOME/<filename>`` or ``~/.local/share/<filename>``

    The parent directory is created if it does not exist.
    """
    if IS_MACOS:
        base = os.path.join(os.path.expanduser("~"), "Library", "Logs")
    elif IS_WINDOWS:
        base = os.environ.get("APPDATA", os.path.expanduser("~"))
    else:
        base = os.environ.get(
            "XDG_STATE_HOME",
            os.path.join(os.path.expanduser("~"), ".local", "share"),
        )
    os.makedirs(base, exist_ok=True)
    return os.path.join(base, filename)


# ── Qt plugin path ────────────────────────────────────────────────────────

def setup_qt_plugin_path(bundle_dir: str) -> None:
    """Set ``QT_PLUGIN_PATH`` for PyInstaller-frozen applications.

    PyInstaller bundles Qt plugins inside the application directory but the
    Qt runtime does not always discover them automatically.  This function
    probes the two most common directory layouts and sets the environment
    variable when a ``platforms`` sub-directory is found.

    Safe to call on any platform — it only mutates the environment when a
    matching plugin directory actually exists.
    """
    for candidate in [
        os.path.join(bundle_dir, "PySide6", "plugins"),
        os.path.join(bundle_dir, "_internal", "PySide6", "plugins"),
    ]:
        if os.path.isdir(os.path.join(candidate, "platforms")):
            os.environ["QT_PLUGIN_PATH"] = candidate
            return
