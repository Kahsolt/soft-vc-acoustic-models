@REM 2022/09/15
@REM Only perform model training
@ECHO OFF

SETLOCAL

SET PYTHON_BIN=python

SET VBANK=%1
SET CONFIG=%2
SET RESUME=%3
IF "%VBANK%"=="" GOTO HELP
IF "%CONFIG%"=="" SET CONFIG=default

SET DATA_PATH=data\%VBANK%
SET LOG_PATH=log\%VBANK%
SET CONFIG_PATH=configs\%CONFIG%.json

IF NOT EXIST %CONFIG_PATH% (
  ECHO ^<^< [Error] missing config file "%CONFIG_PATH%" for `%VBANK%`
  ECHO.
  EXIT /B -1
)

ECHO ^>^> [4/4] training acoustic model to "%LOG_PATH%" (use config "%CONFIG_PATH%")
MKDIR %LOG_PATH%
IF /I "%RESUME%"=="" (
  %PYTHON_BIN% train.py %VBANK% --config %CONFIG_PATH%
) ELSE (
  %PYTHON_BIN% train.py %VBANK% --config %CONFIG_PATH% --resume %RESUME%
)
ECHO.
IF ERRORLEVEL 1 EXIT /B -1

ECHO ^>^> done!

GOTO EOF


:HELP
ECHO Usage: %0 ^<vbank^> ^[config^] ^[resume^]
ECHO   vbank      voice bank name
ECHO   config     config name (default: "default")
ECHO   resume     checkpoint file path to resume from
ECHO.

:EOF
