@echo off
setlocal

set PORT=8501
set APP_DIR=%~dp0

echo [1/2] 检查端口 %PORT% ...
netstat -ano | findstr :%PORT% | findstr LISTENING >nul
if %errorlevel%==0 (
  echo 端口 %PORT% 已在监听，直接打开页面：
  echo http://127.0.0.1:%PORT%
  start http://127.0.0.1:%PORT%
  goto :eof
)

echo [2/2] 未监听，正在启动 Streamlit ...
cd /d "%APP_DIR%"
py -m streamlit run app.py --server.address 127.0.0.1 --server.port %PORT%
