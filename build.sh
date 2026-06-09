#!/usr/bin/env bash
# Build script for TCG Card Centering Detector
#   macOS  → .app bundle with camera permissions → signed → zip
#   Linux  → onedir folder → tar.gz
set -euo pipefail
cd "$(dirname "$0")"

APP_NAME="TCG卡片居中度检测"
VERSION="0.1.5"
PLATFORM="$(uname -s)"

echo "========================================"
echo "  ${APP_NAME} — 打包构建脚本"
echo "  版本: v${VERSION}"
echo "  平台: ${PLATFORM}"
echo "========================================"
echo ""

# ── Clean ─────────────────────────────────────────────────────────────────
if [ -d "dist" ] || [ -d "build" ]; then
    echo "[1/4] 清理旧构建..."
    rm -rf dist build
fi

# ── Dependencies ──────────────────────────────────────────────────────────
echo "[2/4] 检查依赖..."
pip install -q pyinstaller opencv-python numpy pillow pyside6

# ── Build ─────────────────────────────────────────────────────────────────
echo "[3/4] 构建独立应用程序..."

if [ "$PLATFORM" = "Darwin" ]; then
    # ── macOS: spec-driven build (produces .app with camera plist) ────────
    echo "  使用 card_centering.spec (含摄像头权限声明)..."
    pyinstaller --clean card_centering.spec

    APP_BUNDLE="dist/${APP_NAME}.app"

    if [ ! -d "$APP_BUNDLE" ]; then
        echo "  [错误] .app bundle 未生成: ${APP_BUNDLE}"
        exit 1
    fi

    # ── macOS post-build: ad-hoc code signing ─────────────────────────────
    echo ""
    echo "  对 .app 进行 ad-hoc 签名..."
    # Remove Finder quarantine xattr if present (from downloaded deps)
    xattr -cr "$APP_BUNDLE" 2>/dev/null || true
    # Ad-hoc sign all binaries inside the bundle
    codesign --force --deep --sign - "$APP_BUNDLE" 2>&1 || {
        echo "  [警告] 签名失败（不影响运行，但可能触发 Gatekeeper）"
    }

    # Verify the camera privacy key made it into Info.plist
    if plutil -p "${APP_BUNDLE}/Contents/Info.plist" 2>/dev/null | grep -q "NSCameraUsageDescription"; then
        echo "  ✓ 摄像头权限声明已写入 Info.plist"
    else
        echo "  [警告] Info.plist 中未找到摄像头权限声明"
    fi

    # ── Package as zip ────────────────────────────────────────────────────
    echo "[4/4] 打包为独立发行文件..."
    ZIP_NAME="${APP_NAME}_v${VERSION}_macOS.zip"
    # ditto preserves resource forks, symlinks, and code signatures
    ditto -c -k --keepParent "$APP_BUNDLE" "$ZIP_NAME"
    echo "  已创建: ${ZIP_NAME}"

    # Print sizes
    APP_SIZE=$(du -sh "$APP_BUNDLE" 2>/dev/null | cut -f1)
    ZIP_SIZE=$(du -sh "$ZIP_NAME" 2>/dev/null | cut -f1)
    echo ""
    echo "  .app 大小: ${APP_SIZE}"
    echo "  .zip 大小: ${ZIP_SIZE}"

else
    # ── Linux: CLI build (onedir, no .app needed) ─────────────────────────
    pyinstaller --onedir \
        --windowed \
        --name "$APP_NAME" \
        --add-data "src/card_centering:card_centering" \
        --hidden-import cv2 \
        --hidden-import numpy \
        --hidden-import PIL \
        --hidden-import shiboken6 \
        --hidden-import PySide6.QtCore \
        --hidden-import PySide6.QtGui \
        --hidden-import PySide6.QtWidgets \
        --hidden-import PySide6.QtNetwork \
        --hidden-import cv2.data \
        --hidden-import cv2.error \
        --hidden-import cv2.misc \
        --hidden-import cv2.utils \
        --hidden-import numpy._core \
        --hidden-import numpy._core.multiarray \
        --hidden-import numpy._core.umath \
        --hidden-import numpy.linalg \
        --hidden-import numpy.fft \
        --hidden-import numpy.random \
        --exclude-module tkinter \
        --exclude-module matplotlib \
        --exclude-module scipy \
        --exclude-module pytest \
        --clean \
        launcher.py

    # ── Package as tar.gz ─────────────────────────────────────────────────
    echo "[4/4] 打包为独立发行文件..."
    TAR_NAME="${APP_NAME}_v${VERSION}_linux.tar.gz"
    tar -czf "$TAR_NAME" -C dist "$APP_NAME"
    echo "  已创建: ${TAR_NAME}"

    APP_SIZE=$(du -sh "dist/${APP_NAME}" 2>/dev/null | cut -f1)
    TAR_SIZE=$(du -sh "$TAR_NAME" 2>/dev/null | cut -f1)
    echo ""
    echo "  应用大小: ${APP_SIZE}"
    echo "  压缩包大小: ${TAR_SIZE}"
fi

echo ""
echo "========================================"
echo "  构建完成!"
if [ "$PLATFORM" = "Darwin" ]; then
    echo ""
    echo "  安装方法:"
    echo "    1. 解压 ${ZIP_NAME}"
    echo "    2. 将 ${APP_NAME}.app 拖入 /Applications"
    echo "    3. 双击运行（首次需右键→打开以绕过 Gatekeeper）"
    echo ""
    echo "  或直接运行: open \"dist/${APP_NAME}.app\""
else
    echo "  运行: dist/${APP_NAME}/${APP_NAME}"
fi
echo "========================================"
