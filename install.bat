@echo off
title CamAI Attendance - Install

echo ==================================================
echo   CamAI Attendance dependency installer
echo ==================================================
echo.

python --version
echo.

echo [1/3] Upgrading pip...
python -m pip install --upgrade pip
if errorlevel 1 goto :error

echo.
echo [2/3] Installing PyTorch CPU wheels...
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
if errorlevel 1 goto :error

echo.
echo [3/3] Installing project requirements...
python -m pip install -r requirements.txt
if errorlevel 1 goto :error

echo.
echo Checking imports...
python -c "import cv2, numpy, fastapi, sqlalchemy; print('[OK] core packages ready')"
python -c "import faiss; print('[OK] faiss ready')"

echo.
echo Done. Run:
echo   uvicorn src.api.main:app --reload --port 8000
goto :end

:error
echo.
echo Install failed. Check the error above.
exit /b 1

:end
pause
