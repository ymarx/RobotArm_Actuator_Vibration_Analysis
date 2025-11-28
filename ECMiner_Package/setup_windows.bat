@echo off
chcp 65001 >nul
echo ============================================================
echo ECMiner Stage 1 설치 스크립트 (Windows)
echo ============================================================
echo.

:: Python 버전 확인
python --version >nul 2>&1
if errorlevel 1 (
    echo [오류] Python이 설치되어 있지 않습니다.
    echo Python 3.8 이상을 설치한 후 다시 실행하세요.
    echo 다운로드: https://www.python.org/downloads/
    pause
    exit /b 1
)

echo [1/5] Python 버전 확인 중...
python --version
echo.

:: Python 버전 확인 (3.8 이상)
for /f "tokens=2" %%i in ('python --version 2^>^&1') do set PYTHON_VERSION=%%i
for /f "tokens=1,2 delims=." %%a in ("%PYTHON_VERSION%") do (
    set MAJOR=%%a
    set MINOR=%%b
)

if %MAJOR% LSS 3 (
    echo [오류] Python 3.8 이상이 필요합니다. 현재 버전: %PYTHON_VERSION%
    pause
    exit /b 1
)
if %MAJOR% EQU 3 if %MINOR% LSS 8 (
    echo [오류] Python 3.8 이상이 필요합니다. 현재 버전: %PYTHON_VERSION%
    pause
    exit /b 1
)

echo ✓ Python 버전 확인 완료: %PYTHON_VERSION%
echo.

:: 가상 환경 생성
echo [2/5] 가상 환경 생성 중...
if exist "venv" (
    echo [경고] venv 폴더가 이미 존재합니다. 삭제하고 다시 생성합니다.
    rmdir /s /q venv
)

python -m venv venv
if errorlevel 1 (
    echo [오류] 가상 환경 생성에 실패했습니다.
    pause
    exit /b 1
)
echo ✓ 가상 환경 생성 완료
echo.

:: 가상 환경 활성화
echo [3/5] 가상 환경 활성화 중...
call venv\Scripts\activate.bat
if errorlevel 1 (
    echo [오류] 가상 환경 활성화에 실패했습니다.
    pause
    exit /b 1
)
echo ✓ 가상 환경 활성화 완료
echo.

:: pip 업그레이드
echo [4/5] pip 업그레이드 중...
python -m pip install --upgrade pip
echo ✓ pip 업그레이드 완료
echo.

:: 의존성 설치
echo [5/5] 의존성 패키지 설치 중...
pip install -r requirements.txt
if errorlevel 1 (
    echo [오류] 패키지 설치에 실패했습니다.
    pause
    exit /b 1
)
echo ✓ 의존성 설치 완료
echo.

:: 설치 검증
echo ============================================================
echo 설치 검증 중...
echo ============================================================
python -c "import numpy; print('  ✓ NumPy:', numpy.__version__)"
python -c "import pandas; print('  ✓ Pandas:', pandas.__version__)"
python -c "import scipy; print('  ✓ SciPy:', scipy.__version__)"
python -c "import sklearn; print('  ✓ scikit-learn:', sklearn.__version__)"
echo.

echo ============================================================
echo 설치 완료!
echo ============================================================
echo.
echo 다음 단계:
echo   1. data\ 폴더에 원본 CSV 파일을 배치하세요
echo      - data\100W\Sample00_CW.csv, Sample00_CCW.csv, ...
echo      - data\200W\Sample00_CW.csv, Sample00_CCW.csv, ...
echo.
echo   2. ECMiner에서 Python 노드를 생성하고
echo      ecminer_stage1_py_node.py 스크립트를 로드하세요
echo.
echo   3. 가상 환경 활성화 방법:
echo      venv\Scripts\activate
echo.
pause
