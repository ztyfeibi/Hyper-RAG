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
set "SOURCE_CONTEXT=%ROOT%\caches\%SOURCE_DATA_NAME%\contexts\%SOURCE_DATA_NAME%_unique_contexts.json"
set "RUN_MODE=resume"

if /i "%~1"=="fresh" set "RUN_MODE=fresh"

cd /d "%ROOT%" || (
    echo [ERROR] Cannot enter project root: %ROOT%
    pause
    exit /b 1
)

echo ========================================
echo  Hyper-RAG chunk1000 rebuild
echo ========================================
echo        mode=%RUN_MODE%
echo        data=%DATA_NAME%, source=%SOURCE_DATA_NAME%
echo        chunk=%CHUNK_TOKEN_SIZE%, overlap=%CHUNK_OVERLAP_TOKEN_SIZE%, gleaning=%GLEANING%, batch=%BATCH_SIZE%
echo.

if not exist "%PYTHON%" (
    echo [ERROR] Conda python not found:
    echo         %PYTHON%
    pause
    exit /b 1
)

if not exist "%SOURCE_CONTEXT%" (
    echo [ERROR] Source context file not found:
    echo         %SOURCE_CONTEXT%
    pause
    exit /b 1
)

if /i "%RUN_MODE%"=="fresh" (
    if /i not "%TARGET_DIR%"=="%ROOT%\caches\neurology_chunk1000" (
        echo [ERROR] Refuse to delete unexpected target:
        echo         %TARGET_DIR%
        pause
        exit /b 1
    )
    echo [1/3] Fresh rebuild: clearing %TARGET_DIR%
    rmdir /s /q "%TARGET_DIR%" 2>nul
    mkdir "%TARGET_DIR%" 2>nul
) else (
    echo [1/3] Resume rebuild: keeping existing cache files
    if not exist "%TARGET_DIR%" mkdir "%TARGET_DIR%" 2>nul
)

echo.
echo [2/3] Logs
echo        stdout: %TARGET_DIR%\rebuild_stdout.log
echo        stderr: %TARGET_DIR%\rebuild_stderr.log
echo        HyperRAG log: %TARGET_DIR%\HyperRAG.log
echo.
echo [3/3] Start batch rebuild
echo.

"%PYTHON%" reproduce\Step_1.py ^
    --data-name "%DATA_NAME%" ^
    --source-data-name "%SOURCE_DATA_NAME%" ^
    --chunk-token-size "%CHUNK_TOKEN_SIZE%" ^
    --chunk-overlap-token-size "%CHUNK_OVERLAP_TOKEN_SIZE%" ^
    --gleaning "%GLEANING%" ^
    --batch-size "%BATCH_SIZE%" ^
    > "%TARGET_DIR%\rebuild_stdout.log" 2> "%TARGET_DIR%\rebuild_stderr.log"

set "EXIT_CODE=%ERRORLEVEL%"
echo EXIT_CODE=%EXIT_CODE%>> "%TARGET_DIR%\rebuild_stdout.log"

if not "%EXIT_CODE%"=="0" (
    echo.
    echo ========================================
    echo  [FAILED] Rebuild failed with exit code %EXIT_CODE%
    echo  See:
    echo    %TARGET_DIR%\HyperRAG.log
    echo    %TARGET_DIR%\rebuild_stdout.log
    echo    %TARGET_DIR%\rebuild_stderr.log
    echo ========================================
    pause
    exit /b %EXIT_CODE%
)

echo.
echo ========================================
echo  [OK] Rebuild finished
echo  Output: %TARGET_DIR%
echo ========================================
pause
