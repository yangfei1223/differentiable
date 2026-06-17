@echo off
chcp 65001 >nul
setlocal

REM ============================================================
REM  PBR Web Viewer 启动脚本
REM  用法: 双击或命令行执行 start_viewer.bat
REM ============================================================

cd /d "%~dp0"

REM --- 1. 检查 Node.js ---
where node >nul 2>&1
if errorlevel 1 (
    echo [ERROR] 未检测到 Node.js。请先安装 Node.js 18+ (https://nodejs.org/)
    pause
    exit /b 1
)

REM --- 2. 检查 app 目录 ---
if not exist "app" (
    echo [ERROR] 未找到 app 目录，请确认在项目根目录运行
    pause
    exit /b 1
)

REM --- 3. 检查依赖是否已安装 ---
if not exist "app\node_modules" (
    echo [INFO] 首次运行，正在安装依赖...
    cd app
    call npm install
    if errorlevel 1 (
        echo [ERROR] npm install 失败
        pause
        exit /b 1
    )
    cd ..
)

REM --- 4. 检查预置场景资产 ---
if not exist "export\scenes_index.json" (
    echo [WARN] 未找到 export\scenes_index.json
    echo [INFO] 需要先打包场景资产。例如打包 helmet:
    echo   python -m scripts.package_runtime_asset --glb data\helmet_260604\scene\lowpoly.glb --epoch-dir output\helmet_260604_pbr\epoch2000 --scene-name helmet --psnr 20.81
    echo.
)

REM --- 5. 启动 dev server ---
echo [INFO] 启动 Vite dev server...
echo [INFO] 浏览器将自动打开 http://localhost:5173
echo [INFO] 按 Ctrl+C 停止服务
echo.

cd app
call npm run dev

endlocal
