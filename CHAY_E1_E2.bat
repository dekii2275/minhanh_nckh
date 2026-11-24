@echo off
setlocal
cd /d "%~dp0" || exit /b 1
powershell -NoProfile -ExecutionPolicy Bypass -File "src\gt_baselines\run_e1_e2_full.ps1"
set "EXIT_CODE=%ERRORLEVEL%"
echo.
if "%EXIT_CODE%"=="0" (
    echo Da chay xong E1 va E2.
) else (
    echo Chuong trinh dung voi ma loi %EXIT_CODE%.
)
echo Xem log tai outputs\e2_pose_x_retrain_20261124\run_log.txt
pause
exit /b %EXIT_CODE%
