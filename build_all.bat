@echo off
REM Build script for MerkleDB NIF - Run from MSYS2 MinGW64 terminal or cmd with gcc in PATH
setlocal

cd /d C:\Users\baian\merkle_db

set ERL_INCLUDE=C:\Program Files\Erlang OTP\usr\include
set CFLAGS=-O3 -std=c11 -Wall -Wno-unused-variable -Wno-unused-function -Wno-attributes -mavx2
set INCLUDES=-I"%ERL_INCLUDE%" -Inative -Inative\fp_lib\include

echo === Building C Wrapper Objects ===

REM Compile wrapper files
for %%f in (fp_compose fp_correlation_wrappers fp_general_hof fp_generic fp_monads fp_moving_averages_wrappers fp_percentile_wrappers fp_regression_wrappers fp_rolling_window fp_stats_template) do (
    echo Compiling %%f.c...
    gcc %CFLAGS% %INCLUDES% -c native\fp_lib\src\wrappers\%%f.c -o native\%%f.obj
    if errorlevel 1 goto :error
)

REM Compile algorithm files
for %%f in (3d_math_wrapper fp_decision_tree fp_fft fp_gpu_math fp_kmeans fp_lighting fp_linear_regression fp_matrix_ops fp_monte_carlo fp_naive_bayes fp_neural_network fp_pca fp_quaternion_ops fp_radix_sort fp_ray_tracer fp_time_series fp_vector_ops) do (
    echo Compiling %%f.c...
    gcc %CFLAGS% %INCLUDES% -c native\fp_lib\src\algorithms\%%f.c -o native\%%f.obj
    if errorlevel 1 goto :error
)

echo === Linking merkle_nif.dll ===
gcc %CFLAGS% -shared %INCLUDES% -o priv\merkle_nif.dll native\merkle_nif.c native\*.obj
if errorlevel 1 goto :error

echo.
echo === BUILD SUCCESSFUL ===
goto :end

:error
echo.
echo === BUILD FAILED ===
exit /b 1

:end
endlocal
