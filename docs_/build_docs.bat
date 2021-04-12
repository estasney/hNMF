@echo off
CALL ..\venv\Scripts\activate
CALL make clean
CALL make singlehtml
python postprocess.py
git add ..\docs
