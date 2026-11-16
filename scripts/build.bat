@echo off
REM ===================================================================
REM Build script for Traffic Sign Recognition and Road Safety Monitor
REM ===================================================================

if "%JAVA_HOME%"=="" (
    set "JAVA_HOME=C:\Program Files\Java\jdk-25.0.2"
)

set "MVN_CMD=C:\Program Files\Apache NetBeans\java\maven\bin\mvn.cmd"

if not exist "%MVN_CMD%" (
    echo [ERROR] Maven not found at %MVN_CMD%
    echo Please ensure Maven is installed or set on your PATH.
    exit /b 1
)

echo [INFO] Using JAVA_HOME: %JAVA_HOME%
echo [INFO] Running Maven build...
"%MVN_CMD%" %*
