@echo off
REM BLAKE3 NIF Build Script for Windows
REM Requires: GCC (MinGW), Erlang OTP installed

setlocal

set ERL_INCLUDE=C:\Program Files\Erlang OTP\usr\include
set VENDOR=fp_lib\vendor
set OUT=..\priv\blake3_nif.dll

echo === Building BLAKE3 NIF ===
echo.

REM Clean old objects
echo Cleaning old objects...
del /Q *.obj 2>nul

REM Compile BLAKE3 core
echo Compiling BLAKE3 core...
gcc -O3 -mavx2 -msse4.1 -msse2 -DBLAKE3_NO_AVX512 -c %VENDOR%\blake3.c -o blake3.obj -I %VENDOR%
if errorlevel 1 goto :error

REM Compile portable fallback
echo Compiling portable fallback...
gcc -O3 -c %VENDOR%\blake3_portable.c -o blake3_portable.obj -I %VENDOR%
if errorlevel 1 goto :error

REM Compile SIMD dispatch
echo Compiling SIMD dispatch...
gcc -O3 -mavx2 -msse4.1 -msse2 -DBLAKE3_NO_AVX512 -c %VENDOR%\blake3_dispatch.c -o blake3_dispatch.obj -I %VENDOR%
if errorlevel 1 goto :error

REM Compile SSE2
echo Compiling SSE2 kernels...
gcc -O3 -msse2 -c %VENDOR%\blake3_sse2.c -o blake3_sse2.obj -I %VENDOR%
if errorlevel 1 goto :error

REM Compile SSE4.1
echo Compiling SSE4.1 kernels...
gcc -O3 -msse4.1 -c %VENDOR%\blake3_sse41.c -o blake3_sse41.obj -I %VENDOR%
if errorlevel 1 goto :error

REM Compile AVX2
echo Compiling AVX2 kernels...
gcc -O3 -mavx2 -c %VENDOR%\blake3_avx2.c -o blake3_avx2.obj -I %VENDOR%
if errorlevel 1 goto :error

REM Link NIF DLL
echo Linking NIF DLL...
gcc -O3 -shared -mavx2 -I"%ERL_INCLUDE%" -I %VENDOR% -o %OUT% blake3_nif_simple.c blake3.obj blake3_portable.obj blake3_dispatch.obj blake3_sse2.obj blake3_sse41.obj blake3_avx2.obj
if errorlevel 1 goto :error

echo.
echo === Build successful! ===
echo Output: %OUT%
goto :end

:error
echo.
echo === Build FAILED ===
exit /b 1

:end
endlocal
