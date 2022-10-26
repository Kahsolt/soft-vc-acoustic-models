@REM 2022/09/14
@REM Generate all
@ECHO OFF

SETLOCAL ENABLEDELAYEDEXPANSION

SET PYTHON_BIN=python

%PYTHON_BIN% infer.py *

REM %PYTHON_BIN% infer.py ljspeech
REM %PYTHON_BIN% infer.py databaker
REM 
REM %PYTHON_BIN% infer.py aak
REM %PYTHON_BIN% infer.py luoxiaohei
REM %PYTHON_BIN% infer.py vermeil
REM %PYTHON_BIN% infer.py click
REM 
REM %PYTHON_BIN% infer.py sou
REM %PYTHON_BIN% infer.py len
REM %PYTHON_BIN% infer.py lemi
REM %PYTHON_BIN% infer.py hana
REM %PYTHON_BIN% infer.py ema
REM %PYTHON_BIN% infer.py urushi
REM %PYTHON_BIN% infer.py lansi
REM 
REM %PYTHON_BIN% infer.py piano
