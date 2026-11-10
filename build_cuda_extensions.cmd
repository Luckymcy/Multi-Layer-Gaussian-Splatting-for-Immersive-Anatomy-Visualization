@echo off
setlocal EnableExtensions EnableDelayedExpansion

if not defined MLD_VCVARS set "MLD_VCVARS=C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
call "%MLD_VCVARS%"
if errorlevel 1 exit /b %errorlevel%

if not defined MLD_ENV set "MLD_ENV=C:\ProgramData\Anaconda3\envs\mld"
set "CUDA_HOME=%MLD_ENV%"
set "CUDA_PATH=%MLD_ENV%"
set "CUDACXX=%MLD_ENV%\bin\nvcc.exe"
set "DISTUTILS_USE_SDK=1"
if not defined TORCH_CUDA_ARCH_LIST set "TORCH_CUDA_ARCH_LIST=7.5"
if not defined MAX_JOBS set "MAX_JOBS=4"
set "PATH=%MLD_ENV%\bin;%MLD_ENV%\Scripts;!PATH!"

echo CUDA_HOME=%CUDA_HOME%
where cl.exe
where nvcc.exe
where ninja.exe
"%CUDACXX%" --version

"%MLD_ENV%\python.exe" -m pip install --no-build-isolation --verbose "%~dp0Multi-Layer-Anatomy-GS-Training\submodules\diff-gaussian-rasterization"
if errorlevel 1 exit /b %errorlevel%

"%MLD_ENV%\python.exe" -m pip install --no-build-isolation --verbose "%~dp0Multi-Layer-Anatomy-GS-Training\submodules\simple-knn"
if errorlevel 1 exit /b %errorlevel%

"%MLD_ENV%\python.exe" -c "import diff_gaussian_rasterization, simple_knn; print('CUDA extensions imported successfully')"
