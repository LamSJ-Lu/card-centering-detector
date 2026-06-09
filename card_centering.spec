# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for TCG Card Centering Detector.

Build:  pyinstaller card_centering.spec
Output: dist/TCG卡片居中度检测/
"""

import sys
from pathlib import Path

# ---- Project paths ----
PROJECT_ROOT = Path(SPECPATH)
SRC_DIR = PROJECT_ROOT / "src"
PACKAGE_DIR = SRC_DIR / "card_centering"

block_cipher = None

a = Analysis(
    [str(PACKAGE_DIR / "main.py")],
    pathex=[str(SRC_DIR)],
    binaries=[],
    datas=[
        # Include any data files the app might need
    ],
    hiddenimports=[
        # PySide6 hidden imports
        "PySide6.QtCore",
        "PySide6.QtGui",
        "PySide6.QtWidgets",
        "PySide6.QtNetwork",
        "shiboken6",
        # OpenCV
        "cv2",
        "cv2.data",
        "cv2.error",
        "cv2.mat_wrapper",
        "cv2.misc",
        "cv2.utils",
        # numpy
        "numpy",
        "numpy._core",
        "numpy._core.multiarray",
        "numpy._core.umath",
        "numpy.linalg",
        "numpy.fft",
        "numpy.random",
        # PIL
        "PIL",
        "PIL.Image",
        "PIL.ImageTk",
        # Standard library
        "logging",
        "logging.handlers",
        "json",
        "pathlib",
        "collections",
        "dataclasses",
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        # Exclude heavyweight things we don't use
        "tkinter",
        "matplotlib",
        "scipy",
        "pandas",
        "IPython",
        "jupyter",
        "notebook",
        "qtconsole",
        "sphinx",
        "pytest",
        "setuptools",
        "pip",
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name="TCG卡片居中度检测",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False if sys.platform == "darwin" else True,   # UPX fails on macOS ARM64
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=True,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.pure,
    a.datas,
    strip=False,
    upx=False if sys.platform == "darwin" else True,
    upx_exclude=[],
    name="TCG卡片居中度检测",
)

# ── macOS .app bundle ────────────────────────────────────────────────────
# On macOS this produces a proper .app with camera privacy keys in
# Info.plist.  Without NSCameraUsageDescription the OS will deny camera
# access with no helpful error, so the BUNDLE step is mandatory on Darwin.

if sys.platform == "darwin":
    app = BUNDLE(
        coll,
        name="TCG卡片居中度检测.app",
        icon="icon.icns" if Path("icon.icns").exists() else None,
        bundle_identifier="com.cardcentering.detector",
        info_plist={
            "CFBundleName": "TCG卡片居中度检测",
            "CFBundleDisplayName": "TCG Card Centering Detector",
            "CFBundleVersion": "0.1.5",
            "CFBundleShortVersionString": "0.1.5",
            "NSHighResolutionCapable": True,
            "NSCameraUsageDescription": (
                "Camera access is required to capture card photos "
                "for centering analysis."
            ),
        },
    )
