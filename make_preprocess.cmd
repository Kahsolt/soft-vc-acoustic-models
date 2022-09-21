@REM 2022/09/15
@REM Only perform data preprocess
@ECHO OFF

SETLOCAL

SET PYTHON_BIN=python

SET VBANK=%1
SET WAVPATH=%2
IF "%VBANK%"=="" GOTO HELP
IF "%WAVPATH%"=="" GOTO HELP

ECHO ^>^> [0/4] making voicebank `%VBANK%` from "%WAVPATH%"
ECHO.

SET DATA_PATH=data\%VBANK%
SET OUT_PATH=out\%VBANK%
IF NOT EXIST %WAVPATH% (
  ECHO ^<^< [Error] wavpath "%WAVPATH%" does not exist!
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
%PYTHON_BIN% preprocess.py %VBANK% --encode
ECHO.
IF ERRORLEVEL 1 EXIT /B -1

ECHO ^>^> [3/4] prepare melspec ^at "%DATA_PATH%\mels"
MKDIR %DATA_PATH%\mels
%PYTHON_BIN% preprocess.py %VBANK% --melspec
ECHO.
IF ERRORLEVEL 1 EXIT /B -1

ECHO ^>^> done!

GOTO EOF


:HELP
ECHO Usage: %0 ^<vbank^> ^<wavpath^>
ECHO   vbank      voice bank name
ECHO   wavpath    folder path containing *.wav files
ECHO.

:EOF
