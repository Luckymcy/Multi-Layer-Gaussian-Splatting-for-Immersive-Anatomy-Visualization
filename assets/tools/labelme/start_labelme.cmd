@echo off
setlocal
if not defined LABELME_PYTHONW set "LABELME_PYTHONW=C:\ProgramData\Anaconda3\envs\labelme\pythonw.exe"
set "LABELME_WORKSPACE=%~dp0..\..\.."
set "LABELME_IMAGES=%LABELME_WORKSPACE%\data\real\surgery_001\annotation_seed\frames"
set "LABELME_ANNOTATIONS=%LABELME_WORKSPACE%\data\real\surgery_001\annotation_seed\annotations"
if not exist "%LABELME_PYTHONW%" (
    echo Labelme environment was not found.
    pause
    exit /b 1
)
if not exist "%LABELME_IMAGES%" (
    echo Annotation images were not found.
    pause
    exit /b 1
)
if not exist "%LABELME_ANNOTATIONS%" mkdir "%LABELME_ANNOTATIONS%"
start "" "%LABELME_PYTHONW%" -m labelme "%LABELME_IMAGES%" --output "%LABELME_ANNOTATIONS%" --labels fat,exclude --validate-label exact
endlocal
