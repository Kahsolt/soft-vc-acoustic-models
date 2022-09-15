@REM 2022/09/10
@REM Train a new acoustic model from given folder of wavfiles
@ECHO OFF

SETLOCAL

SET PYTHON_BIN=python

SET VBANK=%1
SET WAVPATH=%2
SET CONFIG=%3
IF "%VBANK%"=="" GOTO HELP
IF "%WAVPATH%"=="" GOTO HELP
IF "%CONFIG%"=="" SET CONFIG=default

ECHO ^>^> [0/4] making voicebank `%VBANK%` from "%WAVPATH%"
ECHO.

SET DATA_PATH=data\%VBANK%
SET OUT_PATH=out\%VBANK%
SET CONFIG_PATH=configs\%CONFIG%.json
IF NOT EXIST %WAVPATH% (
  ECHO ^<^< [Error] wavpath "%WAVPATH%" does not exist!
  ECHO.
  EXIT /B -1
)
IF NOT EXIST %CONFIG_PATH% (
  ECHO ^<^< [Error] missing config file "%CONFIG_PATH%" for `%VBANK%`
  ECHO.
  EXIT /B -1
)

ECHO ^>^> [1/4] make workspace "%DATA_PATH%"
MKDIR %DATA_PATH%
ECHO ^>^> link "%WAVPATH%" to "%DATA_PATH%\wavs"
MKLINK /J %DATA_PATH%\wavs %WAVPATH%
ECHO.

ECHO ^>^> [2/4] prepare hubert-unit ^at "%DATA_PATH%\units"
MKDIR %DATA_PATH%\units
%PYTHON_BIN% preprocess.py --encode %DATA_PATH%\wavs %DATA_PATH%\units
ECHO.
IF ERRORLEVEL 1 EXIT /B -1

ECHO ^>^> [3/4] prepare melspec ^at "%DATA_PATH%\mels"
MKDIR %DATA_PATH%\mels
%PYTHON_BIN% preprocess.py --melspec %DATA_PATH%\wavs %DATA_PATH%\mels
ECHO.
IF ERRORLEVEL 1 EXIT /B -1

ECHO ^>^> [4/4] training acoustic model to "%OUT_PATH%" (use config "%CONFIG_PATH%")
MKDIR %OUT_PATH%
%PYTHON_BIN% train.py %DATA_PATH% %OUT_PATH% --config %CONFIG_PATH%
ECHO.
IF ERRORLEVEL 1 EXIT /B -1

ECHO ^>^> done!

GOTO EOF


:HELP
ECHO Usage: %0 ^<vbank^> ^<wavpath^> ^[config^]
ECHO   vbank      voice bank name
ECHO   wavpath    folder path containing *.wav files
ECHO   config     config name (default: "default")
ECHO.
