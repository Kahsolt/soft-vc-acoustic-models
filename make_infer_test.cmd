@REM 2022/09/14 
@REM Generate all
@ECHO OFF

SETLOCAL ENABLEDELAYEDEXPANSION

SET PYTHON_BIN=python

%PYTHON_BIN% infer.py ljspeech test
%PYTHON_BIN% infer.py databaker test
%PYTHON_BIN% infer.py sou test
%PYTHON_BIN% infer.py len test
%PYTHON_BIN% infer.py lemi test
%PYTHON_BIN% infer.py hana test
%PYTHON_BIN% infer.py ema test
%PYTHON_BIN% infer.py urushi test
%PYTHON_BIN% infer.py lansi test
%PYTHON_BIN% infer.py piano test
