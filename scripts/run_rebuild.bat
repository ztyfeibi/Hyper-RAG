@echo off
chcp 65001 >nul
setlocal

set "ROOT=D:\projectes\lyw-rag"
set "PYTHON=D:\Tools\Conda\envs\hyperrag\python.exe"
set "DATA_NAME=neurology_chunk1000"
set "SOURCE_DATA_NAME=neurology"
set "CHUNK_TOKEN_SIZE=1000"
set "CHUNK_OVERLAP_TOKEN_SIZE=150"
set "GLEANING=0"
set "BATCH_SIZE=3000"
set "TARGET_DIR=%ROOT%\caches\%DATA_NAME%"

cd /d "%ROOT%" || exit /b 1
if not exist "%TARGET_DIR%" mkdir "%TARGET_DIR%" 2>nul

"%PYTHON%" reproduce\Step_1.py ^
    --data-name "%DATA_NAME%" ^
    --source-data-name "%SOURCE_DATA_NAME%" ^
    --chunk-token-size "%CHUNK_TOKEN_SIZE%" ^
    --chunk-overlap-token-size "%CHUNK_OVERLAP_TOKEN_SIZE%" ^
    --gleaning "%GLEANING%" ^
    --batch-size "%BATCH_SIZE%" ^
    > "%TARGET_DIR%\rebuild_stdout.log" 2> "%TARGET_DIR%\rebuild_stderr.log"

echo EXIT_CODE=%ERRORLEVEL%>> "%TARGET_DIR%\rebuild_stdout.log"
exit /b %ERRORLEVEL%
