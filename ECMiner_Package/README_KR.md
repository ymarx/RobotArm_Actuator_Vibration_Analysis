# ECMiner Stage 1: 진동 데이터 전처리 및 특징 추출

로봇 암 액추에이터 진동 데이터를 전처리하고 18개 특징을 추출하는 ECMiner Python 노드용 스크립트입니다.

## 📋 목차
- [시스템 요구사항](#시스템-요구사항)
- [설치 방법](#설치-방법)
- [데이터 준비](#데이터-준비)
- [레이블 시스템](#레이블-시스템)
- [ECMiner 연동 방법](#ecminer-연동-방법)
- [하이퍼파라미터](#하이퍼파라미터)
- [성능 지표](#성능-지표)
- [문제 해결](#문제-해결)

---

## 🖥️ 시스템 요구사항

- **운영체제**: Windows 10 이상
- **Python**: 3.8 이상
- **메모리**: 최소 4GB RAM
- **저장공간**: 약 500MB (데이터 포함)

---

## 🚀 설치 방법

### 1. 자동 설치 (권장)

1. `ECMiner_Package` 폴더에서 **PowerShell** 또는 **명령 프롬프트(cmd)** 실행
2. 설치 스크립트 실행:
   ```cmd
   setup_windows.bat
   ```

### 2. 수동 설치

```cmd
# Python 버전 확인 (3.8 이상 필요)
python --version

# 가상 환경 생성
python -m venv venv

# 가상 환경 활성화
venv\Scripts\activate

# pip 업그레이드
python -m pip install --upgrade pip

# 의존성 설치
pip install -r requirements.txt

# 설치 확인
python -c "import numpy, pandas, scipy, sklearn; print('설치 완료')"
```

### 3. 설치 확인

설치가 완료되면 다음 패키지가 설치됩니다:

| 패키지 | 버전 | 용도 |
|--------|------|------|
| NumPy | 1.24.3 | 수치 연산 |
| Pandas | 2.0.3 | 데이터 처리 |
| SciPy | 1.11.1 | 신호 처리 (필터링) |
| scikit-learn | 1.3.0 | 데이터 분할 |
| PyYAML | 6.0+ | 설정 파일 로드 |
| openpyxl | 3.0+ | Excel 파일 읽기 |

---

## 📁 데이터 준비

### 디렉토리 구조

```
ECMiner_Package/
├── ecminer_stage1_py_node.py     ← 독립 실행용 스크립트
├── ecminer_python_node.py        ← ECMiner GUI 연동용 스크립트 (NEW)
├── ecminer_labels.csv            ← 시료별 레이블 정보
├── ecminer_config.yaml           ← 레이블 전략 설정
├── extract_labels_from_excel.py  ← Excel → CSV 변환 스크립트
├── requirements.txt
├── setup_windows.bat
├── README_KR.md
├── data/                         ← 원본 데이터 배치
│   ├── 100W/
│   │   ├── 100W_Sample00 cw4_2025-11-07 03-41-24.csv
│   │   ├── 100W_Sample00 ccw4_2025-11-07 03-39-43.csv
│   │   └── ...
│   └── 200W/
│       ├── 200W_Sample00 cw4_2025-11-07 03-48-46.csv
│       └── ...
└── venv/                         ← 가상 환경
```

### 데이터 파일 형식

**CSV 파일 구조** (7개 채널, 헤더 포함):
```csv
acc-X,acc-Y,acc-Z,acc-Sum,Gyro-X,Gyro-Y,Gyro-Z
0.123,-0.456,0.789,1.234,0.012,-0.034,0.056
...
```

- **샘플링 주파수**: 512 Hz
- **채널**: 가속도 3축 + 합, 자이로 3축
- **파일 명명 규칙**: `{제품}_Sample{번호} {방향}{숫자}_{날짜-시간}.csv`
  - 예: `100W_Sample00 cw4_2025-11-07 03-41-24.csv`
  - 예: `200W_Sample03 ccw4_2025-11-08 14-22-15.csv`

---

## 🏷️ 레이블 시스템

### 개요
ECMiner 패키지는 유연한 레이블 매핑 시스템을 제공하여, "시험전 시료 표기내용.xlsx" 같은 레이블 파일을 기반으로 4단계 레이블(정상/소음/진동/표기없음)을 이진 분류(양품/불량)로 변환합니다.

### 레이블 파일 생성

#### 방법 1: Excel에서 자동 추출 (권장)

1. **Excel 파일 배치**:
   ```
   ECMiner_Package/
   ├── 시험전 시료 표기내용.xlsx  ← 여기에 배치
   └── extract_labels_from_excel.py
   ```

2. **변환 스크립트 실행**:
   ```bash
   cd ECMiner_Package
   python extract_labels_from_excel.py
   ```

3. **결과 확인**:
   ```
   ✓ Excel 파일 발견: 시험전 시료 표기내용.xlsx
   ✓ 레이블 추출 완료: 29개
   ✓ CSV 저장 완료: ecminer_labels.csv
   ```

#### 방법 2: 수동 작성

직접 `ecminer_labels.csv` 파일을 생성할 수도 있습니다:

```csv
product,sample,label
100W,0,정상
100W,1,소음
200W,1,진동
100W,2,진동
200W,2,소음
...
```

### 레이블 파일 (`ecminer_labels.csv`)

자동 또는 수동으로 생성된 시료별 레이블 정보:

- **product**: 제품명 (100W, 200W, ...)
- **sample**: 시료 번호 (0, 1, 2, ..., 20, ...)
- **label**: 레이블 (정상, 소음, 진동, 표기없음)

**중요**:
- `ecminer_stage1_py_node.py`는 같은 폴더의 `ecminer_labels.csv`를 자동으로 참조합니다
- 레이블 정보가 바뀌면 Excel 파일을 업데이트하고 변환 스크립트를 다시 실행하세요
- CSV 파일이 없으면 스크립트 실행 시 오류가 발생합니다

### 설정 파일 (`ecminer_config.yaml`)

레이블 매핑 전략을 정의:

```yaml
label_strategy: "default"  # default, strict, inclusive 중 선택

strategies:
  default:
    description: "소음을 약한 양품으로 포함"
    PASS: ["정상", "소음"]
    FAIL: ["진동", "표기없음"]
    special_overrides:
      - product: "200W"
        sample: 3
        from: "진동"
        to: "정상"
    weights:
      소음: 0.5  # weak positive
```

### 레이블 전략

#### 1. default (기본)
- **양품 (PASS)**: 정상(weight=1.0) + 소음(weight=0.5)
- **불량 (FAIL)**: 진동 + 표기없음
- **특징**: 소음을 약한 양품으로 처리 (학습 시 가중치 절반)

#### 2. strict (엄격)
- **양품 (PASS)**: 정상만
- **불량 (FAIL)**: 소음 + 진동 + 표기없음
- **특징**: 확실한 정상만 양품으로 분류

#### 3. inclusive (포괄)
- **양품 (PASS)**: 정상 + 소음 (둘 다 weight=1.0)
- **불량 (FAIL)**: 진동 + 표기없음
- **특징**: 소음을 정상과 동등하게 처리

### Special Override

특정 시료는 실제 측정 결과에 따라 재분류:
- **200W Sample03**: "진동" → "정상" (Phase1 분석에서 양품 확인)

### 출력 컬럼

레이블 시스템은 3개의 컬럼을 추가로 생성:

| 컬럼 | 타입 | 설명 | 예시 |
|------|------|------|------|
| `label_raw` | str | 원본 레이블 | 정상, 소음, 진동, 표기없음 |
| `label_binary` | int | 이진 레이블 | 1 (양품), 0 (불량) |
| `label_weight` | float | 학습 가중치 | 1.0, 0.5 |

### 전체 워크플로우

#### Step 1: 레이블 파일 생성 (최초 1회 또는 업데이트 시)

```bash
cd ECMiner_Package

# Excel 파일 배치 (같은 폴더 또는 부모 폴더)
# 시험전 시료 표기내용.xlsx

# 레이블 CSV 생성
python extract_labels_from_excel.py

# 결과: ecminer_labels.csv 생성됨
```

#### Step 2: 전략 선택 (선택사항)

```yaml
# ecminer_config.yaml 편집
label_strategy: "default"  # 또는 "strict", "inclusive"
```

#### Step 3: 데이터 전처리 실행

```bash
# ecminer_stage1_py_node.py 실행
python ecminer_stage1_py_node.py

# 자동으로 다음 파일들을 참조:
# - ecminer_labels.csv (레이블 정보)
# - ecminer_config.yaml (레이블 전략)
# - ../ecminer_input_full.csv (입력 파일 목록)
# - ../data/ (원본 데이터)

# 출력: ../ecminer_output_labeled.csv
# 포함 컬럼: label_raw, label_binary, label_weight + 18개 특징
```

### 레이블 업데이트 시

레이블 정보가 변경되면:

1. **Excel 파일 수정**: `시험전 시료 표기내용.xlsx` 편집
2. **CSV 재생성**: `python extract_labels_from_excel.py`
3. **재실행**: `python ecminer_stage1_py_node.py`

기존 `ecminer_labels.csv`가 자동으로 덮어씌워집니다.

### 향후 확장 옵션

**RMS 기반 재분류** (현재 비활성):
```yaml
rms_reclassification:
  enabled: false  # true로 변경하면 활성화
  feature: "acc_Y_rms"
  threshold_method: "normal_range"
  reclassify_to: "정상"
```

Phase1-3 분석 결과, 표기없음의 80.5%가 정상 RMS 범위 내에 위치하므로, 향후 데이터가 충분히 확보되면 RMS 값을 기준으로 표기없음을 재분류할 수 있습니다.

---

## 🔗 실행 방법

ECMiner 패키지는 **두 가지 실행 방식**을 지원합니다:

### 📋 사전 준비: 입력 CSV 생성 (필수)

**모든 실행 방식에서 공통적으로 필요**합니다.

#### 1단계: create_ecminer_input.py 실행

```bash
cd ECMiner_Package

# 가상 환경 활성화 (Windows)
venv\Scripts\activate

# 입력 CSV 생성
python create_ecminer_input.py
```

**출력 결과**:
```
============================================================
ECMiner 입력 CSV 생성 스크립트
============================================================

[1단계] 데이터 폴더 스캔
------------------------------------------------------------
  📁 100W 폴더 스캔 중... → 30개 파일 발견
  📁 200W 폴더 스캔 중... → 30개 파일 발견

[2단계] CSV 파일 생성
------------------------------------------------------------
  ✅ ECMiner 입력 CSV 생성 완료!
     경로: ../ecminer_input_full.csv
     파일 수: 60개
```

**생성 파일**: `ecminer_input_full.csv`
- 위치: 프로젝트 루트 (`ECMiner_Package/..`)
- 형식: `file_path`, `file_id` 컬럼
- 용도: 파일명 정규화 문제 해결 및 고유 ID 생성

**중요**: 이 단계를 건너뛰면 실행 시 오류가 발생합니다!

---

### 방법 1: 독립 실행 (권장 - 테스트/검증용)

`ecminer_stage1_py_node.py`를 사용하여 독립적으로 실행합니다.

#### 실행 방법

```bash
cd ECMiner_Package

# 가상 환경 활성화 (Windows)
venv\Scripts\activate

# 스크립트 실행
python ecminer_stage1_py_node.py

# 출력: ../ecminer_output_labeled.csv
```

**특징**:
- ✅ 상대 경로 자동 처리 (ECMiner_Package 폴더 또는 부모 폴더에서 실행 가능)
- ✅ 입력 CSV 파일 자동 탐색 (ecminer_input_example.csv 우선, ecminer_input_full.csv 차순)
- ✅ 레이블 시스템 자동 적용
- ✅ 디버깅 및 검증에 적합

---

### 방법 2: ECMiner GUI 연동 (권장 - 프로덕션용)

`ecminer_python_node_v2.py`를 ECMiner GUI의 Python 노드에 직접 복사-붙여넣기하여 실행합니다.

#### 2-1. ECMiner Python 노드 생성

1. **ECMiner 실행 후 노드 생성**:
   - 빈 노드 → **Python 연동 노드** 추가
   - 노드 이름: `Stage1_Preprocessing_v2`

2. **스크립트 복사**:
   - `ecminer_python_node_v2.py` 파일 열기
   - **전체 내용 복사** (Ctrl+A → Ctrl+C)
   - ECMiner Python 노드 편집기에 **붙여넣기** (Ctrl+V)

#### 2-2. 경로 설정 (필수)

스크립트 상단의 **🔧 경로 설정** 섹션만 수정:

```python
# ============================================================================
# 🔧 경로 설정 (사용자가 수정하는 영역)
# ============================================================================

# 프로젝트 루트 경로 (절대 경로)
# Windows 예시: "C:/Users/UserName/Projects/RobotArm"
# macOS 예시:   "/Users/UserName/Dropbox/RobotArm"

PROJECT_ROOT = "C:/Users/YourName/Projects/RobotArm_Actuator_QT"

# 출력 CSV 파일 저장 여부
SAVE_OUTPUT_FILE = True  # True: 파일 저장, False: ECMiner로만 전달

# 출력 파일명
OUTPUT_FILENAME = "ecminer_output.csv"
```

**중요**:
- `PROJECT_ROOT`를 **절대 경로**로 설정하세요 (상대 경로 불가)
- 이 경로 아래에 다음이 있어야 합니다:
  - `ecminer_input_full.csv` (create_ecminer_input.py로 생성)
  - `data/` 폴더 (원본 데이터)
  - `ECMiner_Package/` 폴더 (레이블 파일 및 설정)
- Windows: `/` 또는 `\\` 둘 다 사용 가능 (예: `C:/` 또는 `C:\\`)

#### 2-3. 실행

1. **노드 실행**:
   - ECMiner에서 노드 실행 버튼 클릭
   - `ecminer_input_full.csv` 파일 읽기
   - 모든 CSV 파일 처리 시작

2. **출력 확인**:
   - 다음 노드로 `ecmData` 변수 자동 전달
   - `SAVE_OUTPUT_FILE=True`인 경우 파일도 저장

**특징**:
- ✅ **검증된 입력 방식**: ecminer_stage1_py_node.py와 동일한 로직
- ✅ **절대 경로 사용**: ECMiner 임시 폴더에서도 안정적 실행
- ✅ **파일명 정규화 해결**: file_id를 통한 고유성 보장
- ✅ **프로덕션 환경에 최적화**

---

### 방법 비교

| 항목 | 독립 실행 (방법 1) | ECMiner GUI (방법 2) |
|------|-------------------|---------------------|
| **스크립트** | `ecminer_stage1_py_node.py` | `ecminer_python_node_v2.py` |
| **사전 준비** | create_ecminer_input.py 실행 | create_ecminer_input.py 실행 |
| **입력 CSV** | ecminer_input_full.csv | ecminer_input_full.csv |
| **경로 방식** | 상대 경로 (자동 처리) | 절대 경로 (수동 설정) |
| **실행 방식** | 터미널/커맨드 라인 | ECMiner GUI 노드 |
| **용도** | 테스트, 디버깅, 검증 | 프로덕션, 워크플로우 통합 |
| **장점** | 빠른 테스트, 유연한 실행 | 자동화, GUI 통합, 안정적 |

### 워크플로우 요약

```
1. create_ecminer_input.py 실행
   → ecminer_input_full.csv 생성 (60개 파일)

2-A. 독립 실행 (테스트용)
   → python ecminer_stage1_py_node.py
   → ecminer_output_labeled.csv 생성

2-B. ECMiner GUI 실행 (프로덕션용)
   → ecminer_python_node_v2.py 복사-붙여넣기
   → PROJECT_ROOT 수정
   → 실행 → ecmData 출력
```

---

### ECMiner 출력 데이터 형식

| window_id | file_id | dataset_type | label_raw | label_binary | label_weight | product | sample | direction | acc_Y_rms | ... |
|-----------|---------|--------------|-----------|--------------|--------------|---------|--------|-----------|-----------|-----|
| 100W_Sample00_CW_20251107_034124_win000 | 100W_Sample00_CW_20251107_034124 | train | 정상 | 1 | 1.0 | 100W | 0 | CW | 0.123 | ... |
| 200W_Sample01_CCW_20251108_142215_win003 | 200W_Sample01_CCW_20251108_142215 | test | 소음 | 1 | 0.5 | 200W | 1 | CCW | 0.089 | ... |

**출력 컬럼** (28개):
- **메타데이터** (10개):
  - `window_id`: 윈도우 고유 ID (타임스탬프 포함)
  - `file_id`: 파일 고유 ID (타임스탬프 포함)
  - `dataset_type`: train/val/test
  - `label_raw`: 원본 레이블 (정상/소음/진동/표기없음)
  - `label_binary`: 이진 레이블 (1=양품, 0=불량)
  - `label_weight`: 학습 가중치 (1.0 또는 0.5)
  - `product`: 제품명 (100W/200W)
  - `sample`: 시료 번호 (0, 1, 2, ...)
  - `direction`: 회전 방향 (CW/CCW)
  - `timestamp`: 측정 시간 (YYYYMMDD_HHMMSS)

- **기본 특징** (9개):
  - acc_Y_rms, acc_Y_peak, acc_Y_crest
  - acc_Sum_rms, acc_Sum_peak, acc_Sum_crest
  - Gyro_Y_rms, Gyro_Y_peak, Gyro_Y_crest

- **밴드 RMS 특징** (9개):
  - acc_Y_rms_low, acc_Y_rms_mid, acc_Y_rms_high
  - acc_Sum_rms_low, acc_Sum_rms_mid, acc_Sum_rms_high
  - Gyro_Y_rms_low, Gyro_Y_rms_mid, Gyro_Y_rms_high

---

## ⚙️ 하이퍼파라미터

### 전처리 파라미터

| 파라미터 | 값 | 설명 |
|----------|-----|------|
| `SAMPLING_FREQ` | 512.0 Hz | 샘플링 주파수 |
| `WINDOW_SEC` | 8.0 초 | 윈도우 크기 |
| `HOP_SEC` | 4.0 초 | Hop 크기 (50% 중첩) |
| `STABLE_MARGIN` | 0.1 (10%) | 앞/뒤 안정 구간 제외 비율 |
| `RANDOM_SEED` | 42 | 재현성을 위한 난수 시드 |

### 데이터 분할 파라미터

**양품 파일 (시간 기반 분할)**:
| Split | 범위 | 비율 |
|-------|------|------|
| Train | 0-60% | 60% |
| Val | 60-80% | 20% |
| Test | 80-100% | 20% |

**불량 파일 (파일 기반 분할)**:
| Split | 비율 |
|-------|------|
| Train | 70% |
| Val | 15% |
| Test | 15% |

### 주파수 밴드 파라미터

| 밴드 | 주파수 범위 | 물리적 의미 |
|------|-------------|-------------|
| Low | 1-10 Hz | 구조 진동 |
| Mid | 10-50 Hz | 회전 고조파 |
| High | 50-150 Hz | 베어링 결함, 충격 |

### XGBoost 모델 파라미터 (참고용)

**ECMiner 4개 파라미터** (ECMiner에서 조정 가능):
| 파라미터 | 값 | 설명 |
|----------|-----|------|
| `n_estimators` | 100 | 트리 개수 |
| `max_depth` | 3 | 트리 최대 깊이 |
| `subsample` | 0.8 | 샘플 서브샘플링 비율 |
| `learning_rate` | 0.1 | 학습률 |

**Phase3-1 전체 파라미터** (Python 환경 참고용):
| 파라미터 | 값 | 설명 |
|----------|-----|------|
| `n_estimators` | 100 | 트리 개수 |
| `max_depth` | 3 | 트리 최대 깊이 |
| `subsample` | 0.8 | 샘플 서브샘플링 비율 |
| `learning_rate` | 0.1 | 학습률 |
| `min_child_weight` | 1 | 최소 자식 가중치 |
| `reg_lambda` | 1.0 | L2 정규화 |
| `reg_alpha` | 0.0 | L1 정규화 |
| `colsample_bytree` | 1.0 | 컬럼 서브샘플링 비율 |
| `scale_pos_weight` | 자동 | 클래스 불균형 처리 |

---

## 📊 성능 지표

### ECMiner 환경 성능 (4개 파라미터만 사용)

| 특징 세트 | Test AUC | Train/Val/Test 윈도우 수 |
|-----------|----------|--------------------------|
| **18개 특징** (전체) | **0.9882** | 455/102/104 |
| **12개 특징** (핵심만) | **0.9764** | 455/102/104 |

**12개 핵심 특징**:
- acc_Y_rms, acc_Y_peak, acc_Y_crest
- acc_Sum_rms, acc_Sum_peak, acc_Sum_crest
- Gyro_Y_rms, Gyro_Y_peak, Gyro_Y_crest
- acc_Y_rms_low, acc_Sum_rms_low, Gyro_Y_rms_low

### 클래스 분포 (60개 파일, 661 윈도우)

| Dataset | Normal | Abnormal | Total |
|---------|--------|----------|-------|
| Train | 26 | 429 | 455 |
| Val | 8 | 94 | 102 |
| Test | 7 | 97 | 104 |
| **Total** | **41** | **620** | **661** |

**데이터 특성**:
- ✅ **파일 고유성 보장**: 타임스탬프 포함 file_id로 중복 제거
- ✅ **시간 기반 분할**: Normal 파일(4개)은 시간순 train/val/test 분할
- ✅ **파일 단위 분할**: Abnormal 파일(56개)은 파일 단위 7:1.5:1.5 분할
- ✅ **데이터 무결성**: window_id 중복 없음, 데이터 누수 없음

**해석**:
- ✅ **높은 성능 (Test AUC 0.98)**: 특징이 discriminative하고 분할이 적절함
- ✅ **12개 핵심 특징**: 18개 → 12개 특징으로도 AUC 0.97 유지
- ✅ **ECMiner 4개 파라미터**: 제한된 파라미터로도 우수한 성능
- ⚠️ **클래스 불균형**: Normal 6%, Abnormal 94% → scale_pos_weight 16.5

---

## 🐛 문제 해결

### 설치 문제

**1. Python이 인식되지 않음**
```
'python'은(는) 내부 또는 외부 명령... 이 아닙니다.
```
**해결**: Python이 PATH에 등록되어 있는지 확인
- Python 재설치 시 "Add Python to PATH" 옵션 선택

**2. venv 모듈을 찾을 수 없음**
```
No module named venv
```
**해결**: Python 전체 설치 필요 (Microsoft Store 버전은 불완전할 수 있음)

**3. 패키지 설치 실패**
```
ERROR: Could not find a version that satisfies the requirement...
```
**해결**: pip 업그레이드 후 재시도
```cmd
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 실행 문제

**1. 파일을 찾을 수 없음**
```
FileNotFoundError: [Errno 2] No such file or directory: 'data/100W/Sample00_CW.csv'
```
**해결**:
- `data/` 폴더 구조 확인
- CSV 파일이 올바른 위치에 있는지 확인
- 파일 경로가 상대 경로로 올바른지 확인

**2. 입력 컬럼 누락**
```
ValueError: 입력 데이터에 'file_path' 컬럼이 없습니다.
```
**해결**: ECMiner 입력 CSV에 필수 컬럼 (file_path, file_id, label) 포함 확인

**3. 메모리 부족**
```
MemoryError: Unable to allocate...
```
**해결**:
- 한 번에 처리하는 파일 수 줄이기
- 메모리 4GB 이상 권장

**4. NaN 값 경고**
```
경고: 결과에 NaN 값이 포함되어 있습니다.
```
**해결**:
- 원본 CSV 파일의 결측치 확인
- 특정 채널 데이터가 모두 누락되었는지 확인

### ECMiner 연동 문제

**1. PROJECT_ROOT 경로 오류**
```
FileNotFoundError: [Errno 2] No such file or directory: '/path/to/data/100W'
```
**해결**:
- `ecminer_python_node.py` 상단의 `PROJECT_ROOT` 확인
- 절대 경로로 올바르게 설정되었는지 확인
- 경로 끝에 `/` 또는 `\` 없이 설정
- Windows: `C:/Users/...` 또는 `C:\\Users\\...`
- macOS/Linux: `/Users/...` 또는 `/home/...`

**2. 레이블 파일 없음**
```
FileNotFoundError: 레이블 파일을 찾을 수 없습니다: .../ecminer_labels.csv
```
**해결**:
- `ECMiner_Package/ecminer_labels.csv` 파일 존재 확인
- Excel 파일에서 레이블 추출: `python extract_labels_from_excel.py`

**3. 설정 파일 없음**
```
FileNotFoundError: 설정 파일을 찾을 수 없습니다: .../ecminer_config.yaml
```
**해결**:
- `ECMiner_Package/ecminer_config.yaml` 파일 존재 확인
- 파일이 없으면 README의 레이블 시스템 섹션 참조하여 생성

**4. 독립 실행 시 경로 문제**
```
FileNotFoundError: INPUT CSV 파일을 찾을 수 없습니다
```
**해결** (`ecminer_stage1_py_node.py` 전용):
- `ECMiner_Package` 폴더에서 실행하는지 확인
- `ecminer_input_example.csv` 또는 `ecminer_input_full.csv` 존재 확인
- 부모 폴더에도 자동으로 탐색하므로 부모 폴더에 배치 가능

**5. ECMiner GUI 임시 폴더 문제**
- ECMiner는 임시 폴더에서 스크립트를 실행합니다
- **반드시** `ecminer_python_node.py` 사용 (절대 경로 방식)
- `ecminer_stage1_py_node.py`는 독립 실행 전용 (상대 경로 방식)

**6. 출력 데이터 형식 불일치**
- 출력 CSV의 컬럼 순서 확인
- ECMiner가 `ecmData` 변수를 올바르게 인식하는지 확인
- `SAVE_OUTPUT_FILE=True`로 설정하여 파일로도 저장 후 검증

---

## 📞 지원 및 문의

- **프로젝트 위치**: `/Users/YMARX/Dropbox/2025_ECMiner/CP25_NeuroMecha/03_진행/[Analysis]RobotArm_Actuator_QT(Ociliation)/`
- **문서**: `docs/` 폴더에서 상세 분석 보고서 참조
- **데이터**: 원본 데이터는 `data/100W/`, `data/200W/` 폴더에 보관

---

## 📝 라이선스 및 저작권

이 스크립트는 로봇 암 액추에이터 진동 분석 프로젝트의 일부입니다.

**개발 정보**:
- 프로젝트: Robot Arm Actuator Vibration Analysis (Phase 3)
- 기간: 2025-11-17 ~ 2025-11-24
- 최종 버전: ECMiner Stage 1 (2025-11-24)

---

## 🔄 버전 히스토리

### v1.2 (2025-11-25) - ECMiner GUI 통합
- ✅ **NEW**: `ecminer_python_node.py` - ECMiner GUI 전용 스크립트
  - 절대 경로 방식으로 임시 폴더 실행 지원
  - 폴더 자동 스캔 (입력 CSV 불필요)
  - 복사-붙여넣기 간편 실행
- ✅ **IMPROVED**: `ecminer_stage1_py_node.py` - 독립 실행 최적화
  - 상대 경로 자동 처리 (ECMiner_Package 또는 부모 폴더)
  - 입력 CSV 다중 위치 탐색
  - Raw data 경로 자동 탐색
- ✅ 두 가지 실행 방식 지원 (독립 실행 vs ECMiner GUI)
- ✅ 레이블 시스템 완전 통합 (Excel → CSV 자동 변환)
- ✅ 타임스탬프 포함 file_id/window_id (데이터 고유성 보장)
- ✅ 독립 실행 가이드 문서 추가 (STANDALONE_EXECUTION_GUIDE.md)

### v1.0 (2025-11-24) - 초기 릴리스
- ✅ ECMiner Python 노드 연동 지원
- ✅ 18개 특징 추출 (9개 기본 + 9개 밴드 RMS)
- ✅ 시간 기반 분할 (양품) + 파일 기반 분할 (불량)
- ✅ 데이터 무결성 검증 (중복 제거)
- ✅ 한국어 주석 및 문서화
- ✅ Windows CLI 설치 스크립트

---

**Happy Analyzing! 🚀**
