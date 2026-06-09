@echo off
chcp 65001 >nul
cd /d "%~dp0"

echo ========================================
echo   TCG 卡片居中度检测 — 打包构建脚本
echo ========================================
echo.

:: Clean previous build
if exist "dist" (
    echo [1/4] 清理旧构建...
    rmdir /s /q "dist" 2>nul
    rmdir /s /q "build" 2>nul
)

:: Install dependencies if needed
echo [2/4] 检查依赖...
pip install -q pyinstaller opencv-python numpy pillow pyside6 2>nul

:: Build
echo [3/4] 构建独立可执行文件...
pyinstaller --onedir ^
    --windowed ^
    --name "TCG卡片居中度检测" ^
    --add-data "src/card_centering;card_centering" ^
    --hidden-import cv2 ^
    --hidden-import numpy ^
    --hidden-import PIL ^
    --hidden-import shiboken6 ^
    --hidden-import PySide6.QtCore ^
    --hidden-import PySide6.QtGui ^
    --hidden-import PySide6.QtWidgets ^
    --hidden-import PySide6.QtNetwork ^
    --exclude-module tkinter ^
    --exclude-module matplotlib ^
    --exclude-module scipy ^
    --exclude-module pytest ^
    --clean ^
    launcher.py

if errorlevel 1 (
    echo.
    echo [错误] 构建失败!
    pause
    exit /b 1
)

:: Package as ZIP
echo [4/4] 打包发行文件...
powershell -NoProfile -Command ^
    "Compress-Archive -Path 'dist/TCG卡片居中度检测/*' -DestinationPath 'TCG卡片居中度检测_v0.1.0.zip' -Force"

echo.
echo ========================================
echo   构建完成!
echo   输出目录: dist\TCG卡片居中度检测\
echo   压缩包:   TCG卡片居中度检测_v0.1.0.zip
echo ========================================
echo.
pause
