@echo off
setlocal

set "PROJECT_DIR=%~dp0"
set "PROJECT_DIR=%PROJECT_DIR:~0,-1%"
set "SCRIPTS_EXE=%PROJECT_DIR%\..\..\cmake-build-debug\dependencies\vglib.github.io\scripts\Debug\scripts.exe"

if not exist "%SCRIPTS_EXE%" (
    set "SCRIPTS_EXE=%PROJECT_DIR%\cmake-build-debug\scripts\Debug\scripts.exe"
)

echo compiling glsl shaders...
python "%PROJECT_DIR%\scripts\compile_shaders.py" "%PROJECT_DIR%\data\shaders"

echo generating cpp header / source file including all shaders...
"%SCRIPTS_EXE%" "%PROJECT_DIR%\data" "%PROJECT_DIR%\VulkanBase\include" "%PROJECT_DIR%\VulkanBase\source" glsl_shaders "%PROJECT_DIR%"
