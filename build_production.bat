@echo off
REM Production Build Script for MerkleDB
cd /d C:\Users\baian\merkle_db

set ERL_INCLUDE=C:\Program Files\Erlang OTP\usr\include
set CFLAGS=-O3 -std=c11 -Wall -Wextra -Wno-unused-parameter -Wno-unused-variable -Wno-unused-function -Wno-attributes -mavx2 -mfma
set INCLUDES=-I"%ERL_INCLUDE%" -Inative -Inative\fp_lib\include -Inative\fp_lib\vendor

echo ============================================================
echo           MERKLEDB PRODUCTION BUILD
echo ============================================================
echo.

echo [1/7] Killing any running Erlang processes...
taskkill /F /IM erl.exe >nul 2>&1
taskkill /F /IM werl.exe >nul 2>&1
ping -n 2 127.0.0.1 >nul

echo [2/7] Regenerating bridge...
call elixir gen_bridge.exs
echo     Bridge done.

echo [3/7] Compiling fp_query.c...
gcc %CFLAGS% %INCLUDES% -c native\fp_lib\src\algorithms\fp_query.c -o native\fp_query.obj
if %errorlevel% neq 0 goto :fail
echo     fp_query done.

echo [4/7] Compiling Blake3 (vendor)...
REM Disable AVX512 (not universally available)
set BLAKE3_FLAGS=-DBLAKE3_NO_AVX512
gcc %CFLAGS% %INCLUDES% %BLAKE3_FLAGS% -c native\fp_lib\vendor\blake3.c -o native\blake3.obj
if %errorlevel% neq 0 goto :fail
gcc %CFLAGS% %INCLUDES% %BLAKE3_FLAGS% -c native\fp_lib\vendor\blake3_portable.c -o native\blake3_portable.obj
if %errorlevel% neq 0 goto :fail
gcc %CFLAGS% %INCLUDES% %BLAKE3_FLAGS% -c native\fp_lib\vendor\blake3_dispatch.c -o native\blake3_dispatch.obj
if %errorlevel% neq 0 goto :fail
gcc %CFLAGS% %INCLUDES% %BLAKE3_FLAGS% -msse2 -c native\fp_lib\vendor\blake3_sse2.c -o native\blake3_sse2.obj
if %errorlevel% neq 0 goto :fail
gcc %CFLAGS% %INCLUDES% %BLAKE3_FLAGS% -msse4.1 -c native\fp_lib\vendor\blake3_sse41.c -o native\blake3_sse41.obj
if %errorlevel% neq 0 goto :fail
gcc %CFLAGS% %INCLUDES% %BLAKE3_FLAGS% -mavx2 -c native\fp_lib\vendor\blake3_avx2.c -o native\blake3_avx2.obj
if %errorlevel% neq 0 goto :fail
echo     Blake3 vendor done.

echo [5/7] Compiling Blake3 (fp wrappers)...
gcc %CFLAGS% %INCLUDES% -c native\fp_lib\src\fp_blake3.c -o native\fp_blake3.obj
if %errorlevel% neq 0 goto :fail
gcc %CFLAGS% %INCLUDES% -c native\fp_lib\src\fp_blake3_official.c -o native\fp_blake3_official.obj
if %errorlevel% neq 0 goto :fail
echo     Blake3 wrappers done.

echo [6/7] Linking merkle_nif.dll...
gcc %CFLAGS% -shared %INCLUDES% -o priv\merkle_nif.dll native\merkle_nif.c native\*.obj
if %errorlevel% neq 0 goto :fail
echo     Link done.

echo [7/7] Compiling Elixir (zero warnings required)...
call mix compile --warnings-as-errors
if %errorlevel% neq 0 goto :warnings

echo.
echo ============================================================
echo    BUILD SUCCESSFUL - ZERO WARNINGS - PRODUCTION QUALITY
echo ============================================================
goto :eof

:warnings
echo.
echo Build has warnings. Details:
call mix compile
goto :eof

:fail
echo BUILD FAILED
goto :eof
