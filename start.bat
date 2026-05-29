@echo off
chcp 65001 >nul
echo ==========================================
echo  淘天播术-电商直播话术军师 - 直播话术智能分析系统
echo ==========================================
echo.

REM 检查Python是否安装
python --version >nul 2>&1
if errorlevel 1 (
    echo [错误] 未检测到Python，请先安装Python 3.8或更高版本
    pause
    exit /b 1
)

echo [1/3] 正在检查依赖...
REM 检查并安装依赖
pip install -q -r requirements.txt
if errorlevel 1 (
    echo [警告] 部分依赖安装失败，尝试继续启动...
)

echo [2/3] 正在启动系统...
echo.
echo 系统将在浏览器中打开，请稍候...
echo 如果浏览器没有自动打开，请手动访问：http://localhost:8501
echo.
echo 按Ctrl+C可以停止服务
echo.

REM 启动Streamlit
cd /d "%~dp0"
streamlit run app.py --server.port=8501 --server.address=localhost

pause
