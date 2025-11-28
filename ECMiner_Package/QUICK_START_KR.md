# ECMiner Stage 1 빠른 시작 가이드 v2

> **새로운 하이브리드 방식**: 검증된 입력 + 절대 경로 = 안정적 실행

---

## 📦 설치 (3분)

### Windows 사용자
```cmd
1. ECMiner_Package 폴더 열기
2. setup_windows.bat 더블클릭
3. 설치 완료 대기 (약 2-3분)
```

### 수동 설치
```cmd
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

---

## 📁 데이터 배치

```
프로젝트_루트/
├── ECMiner_Package/
│   ├── create_ecminer_input.py      ← ✨ NEW: 입력 CSV 생성
│   ├── ecminer_stage1_py_node.py    ← 독립 실행용
│   ├── ecminer_python_node_v2.py    ← ✨ NEW: ECMiner GUI용 v2
│   ├── ecminer_labels.csv           ← 레이블 파일
│   └── ecminer_config.yaml          ← 설정 파일
├── ecminer_input_full.csv           ← 생성될 입력 파일
└── data/
    ├── 100W/
    │   ├── 100W_Sample00 cw4_2025-11-07 03-41-24.csv
    │   └── ...
    └── 200W/
        ├── 200W_Sample00 cw4_2025-11-07 03-48-46.csv
        └── ...
```

---

## 🚀 실행 방법

### 📋 STEP 0: 입력 CSV 생성 (필수 - 최초 1회)

**모든 실행 방식에서 공통적으로 필요**합니다.

```bash
cd ECMiner_Package
venv\Scripts\activate  # Windows
python create_ecminer_input.py
```

**결과**:
```
✅ ECMiner 입력 CSV 생성 완료!
   경로: ../ecminer_input_full.csv
   파일 수: 60개
```

**중요**: 파일명 정규화 문제를 해결하고 고유 ID를 생성합니다!

---

### 방법 A: 독립 실행 (테스트/검증용)

**빠른 테스트에 적합**

```bash
cd ECMiner_Package
venv\Scripts\activate
python ecminer_stage1_py_node.py
```

- ✅ 입력 CSV: `ecminer_input_full.csv` 자동 탐색
- ✅ 상대 경로 자동 처리
- ✅ 디버깅 용이

**출력**: `../ecminer_output_labeled.csv`

---

### 방법 B: ECMiner GUI 연동 (프로덕션용)

**프로덕션 워크플로우에 적합**

#### 1단계: 노드 생성 및 스크립트 복사

1. ECMiner에서 **Python 연동 노드** 추가
2. `ecminer_python_node_v2.py` 파일 열기
3. **전체 내용 복사** → ECMiner 편집기에 **붙여넣기**

#### 2단계: 경로 설정 (필수)

스크립트 상단에서 `PROJECT_ROOT`만 수정:

```python
# Windows
PROJECT_ROOT = "C:/Users/YourName/Projects/RobotArm"

# macOS/Linux
PROJECT_ROOT = "/Users/YourName/Dropbox/RobotArm"
```

**필수 파일 확인**:
- `{PROJECT_ROOT}/ecminer_input_full.csv` (create_ecminer_input.py로 생성)
- `{PROJECT_ROOT}/data/100W/`, `{PROJECT_ROOT}/data/200W/`
- `{PROJECT_ROOT}/ECMiner_Package/ecminer_labels.csv`

#### 3단계: 실행

- ECMiner 노드 실행 버튼 클릭
- `ecminer_input_full.csv` 파일 읽기
- 모든 CSV 파일 처리 시작
- 결과: 다음 노드로 `ecmData` 변수 전달

**특징**:
- ✅ 검증된 입력 방식 (ecminer_stage1_py_node.py와 동일 로직)
- ✅ 절대 경로 방식 (임시 폴더 실행 안정적)
- ✅ 파일명 정규화 해결 (file_id를 통한 고유성 보장)

---

## 📊 출력 형식

**출력 컬럼** (28개):
- **메타데이터** (10개): window_id, file_id, dataset_type, label_raw, label_binary, label_weight, product, sample, direction, timestamp
- **기본 특징** (9개): acc_Y_rms, acc_Y_peak, acc_Y_crest, ...
- **밴드 RMS** (9개): acc_Y_rms_low, acc_Y_rms_mid, acc_Y_rms_high, ...

---

## ⚡ 성능

| 특징 수 | Test AUC | 추천 |
|---------|----------|------|
| 18개 (전체) | 0.9882 | ✅ 최고 성능 |
| 12개 (핵심) | 0.9764 | ✅ 해석력 우선 |

**12개 핵심 특징**:
- acc_Y: rms, peak, crest
- acc_Sum: rms, peak, crest
- Gyro_Y: rms, peak, crest
- 밴드 RMS: acc_Y_rms_low, acc_Sum_rms_low, Gyro_Y_rms_low

---

## 🐛 문제 발생 시

### STEP 0을 건너뛴 경우
```
❌ FileNotFoundError: ecminer_input_full.csv
```
→ `python create_ecminer_input.py` 실행

### Python 인식 안 됨
→ Python 3.8+ 설치, PATH 등록 확인

### ECMiner GUI: PROJECT_ROOT 경로 오류
→ `ecminer_python_node_v2.py` 상단의 `PROJECT_ROOT`를 절대 경로로 설정

### 독립 실행: 파일 없음 오류
→ `ECMiner_Package` 폴더에서 실행, `data/` 폴더 구조 확인

### 레이블 파일 없음
→ `python extract_labels_from_excel.py` 실행하여 레이블 생성

### 파일명 파싱 실패 경고
→ `create_ecminer_input.py`가 자동으로 정규화 처리, 무시 가능

**상세 가이드**: `README_KR.md` 참조

---

## 📋 워크플로우 요약

```
1️⃣ create_ecminer_input.py 실행 (최초 1회)
   → ecminer_input_full.csv 생성

2️⃣ 방법 선택:
   A. 독립 실행: python ecminer_stage1_py_node.py
   B. ECMiner GUI: ecminer_python_node_v2.py 복사-붙여넣기

3️⃣ 실행 및 결과 확인
   → 661개 윈도우 생성 (60개 파일)
   → Train: 485, Val: 84, Test: 92
```

---

## 📞 더 자세한 내용

- **전체 설명서**: `README_KR.md`
- **새로운 스크립트**: `create_ecminer_input.py`, `ecminer_python_node_v2.py`
- **기존 스크립트 (호환)**: `ecminer_stage1_py_node.py`, `ecminer_python_node.py`
- **레이블 시스템**: `ecminer_labels.csv`, `ecminer_config.yaml`

**Happy Analyzing! 🚀**
