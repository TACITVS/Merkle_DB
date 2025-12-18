@echo off
setlocal enabledelayedexpansion

echo [1/3] Generating Bridge Code...
call elixir gen_bridge.exs
if %errorlevel% neq 0 exit /b %errorlevel%

echo [2/3] Compiling Assembly Library...
set OBJ_FILES=

:: Ensure the ASM directory exists
if not exist "native\fp_lib\src\asm\" (
    echo [ERROR] Directory native\fp_lib\src\asm\ not found!
    exit /b 1
)

:: Loop through every .asm file
:: UPDATED: Added -Inative\fp_lib\src\asm\ so NASM finds 'macros.inc'
for %%f in (native\fp_lib\src\asm\*.asm) do (
    echo    - Compiling %%~nxf...
    
    nasm -f win64 -Inative\fp_lib\src\asm\ "%%~ff" -o "native\%%~nf.obj"
    
    if !errorlevel! neq 0 (
        echo [ERROR] Failed to compile %%~nxf
        exit /b 1
    )
    set OBJ_FILES=!OBJ_FILES! "native\%%~nf.obj"
)

echo [3/3] Compiling C NIF (Linking All Objects)...
gcc -O3 -std=c11 -shared ^
    -I"C:\Program Files\Erlang OTP\erts-14.0.1\include" ^
    -Inative ^
    -Inative/fp_lib/include ^
    -o priv/merkle_nif.dll ^
    native/merkle_nif.c !OBJ_FILES!

if %errorlevel% neq 0 (
    echo [ERROR] GCC compilation failed.
    exit /b %errorlevel%
)

echo [SUCCESS] Build Complete.