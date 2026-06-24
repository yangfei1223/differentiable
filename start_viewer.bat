@echo off
cd /d "%~dp0"

REM ============================================================
REM  PBR Web Viewer Launcher
REM ============================================================

REM --- 1. Check Node.js ---
where node >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Node.js not found. Install Node.js 18+
    pause
    exit /b 1
)

REM --- 2. Check app dir ---
if not exist "app\" (
    echo [ERROR] app/ not found. Run from project root.
    pause
    exit /b 1
)

REM --- 3. First-run dependency install ---
if not exist "app\node_modules\" (
    echo [INFO] Installing dependencies...
    cd app
    call npm install
    if %errorlevel% neq 0 (
        echo [ERROR] npm install failed
        cd ..
        pause
        exit /b 1
    )
    cd ..
)

REM --- 4. Warn if no pre-packed scene assets ---
if not exist "export\scenes_index.json" (
    echo [WARN] export\scenes_index.json missing.
    echo [INFO] Pack a scene first:
    echo   python -m scripts.package_runtime_asset --glb data\helmet_260604\scene\lowpoly.glb --epoch-dir output\helmet_260604_pbr\epoch2000 --scene-name helmet
    echo.
)

REM --- 5. Start dev server ---
echo [INFO] Starting Vite dev server...
echo.
cd app
call npm run dev
