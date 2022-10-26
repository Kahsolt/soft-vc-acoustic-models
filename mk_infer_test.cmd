@REM 2022/09/14
@REM Generate all
@ECHO OFF

SETLOCAL ENABLEDELAYEDEXPANSION

SET PYTHON_BIN=python

rem %PYTHON_BIN% infer.py *

%PYTHON_BIN% infer.py default
%PYTHON_BIN% infer.py ljspeech
%PYTHON_BIN% infer.py databaker

%PYTHON_BIN% infer.py aak
%PYTHON_BIN% infer.py luoxiaohei
%PYTHON_BIN% infer.py vermeil
%PYTHON_BIN% infer.py click

%PYTHON_BIN% infer.py sou
%PYTHON_BIN% infer.py len
%PYTHON_BIN% infer.py lemi
%PYTHON_BIN% infer.py hana
%PYTHON_BIN% infer.py ema
%PYTHON_BIN% infer.py urushi
%PYTHON_BIN% infer.py lansi

%PYTHON_BIN% infer.py piano
