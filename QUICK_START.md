# 🚀 빠른 시작 가이드

로봇팔 액추에이터 진동 분석 프로젝트를 빠르게 시작하기 위한 단계별 가이드입니다.

---

## 📋 사전 요구사항

- Python 3.10 이상
- 원본 데이터: `100W/`, `200W/` 폴더 및 `시험전 시료 표기내용.xlsx`

---

## ⚡ 5분 시작

### 1. 환경 설정

```bash
# 가상환경 생성
python -m venv venv

# 가상환경 활성화
source venv/bin/activate  # macOS/Linux
# venv\Scripts\activate    # Windows

# 의존성 설치
pip install -r requirements.txt
```

### 2. 파이프라인 실행

```bash
# 전체 파이프라인 실행 (약 2-5분 소요)
python run_pipeline.py
```

### 3. 결과 확인

파이프라인 완료 후 생성되는 파일:

```
data/
├── interim/
│   ├── file_master_v1.parquet    # 파일 메타데이터 + 라벨
│   ├── quality_report.csv        # 품질 검사 결과
│   ├── windows_meta_v1.parquet   # 윈도우 세그먼트 정보
│   └── splits/
│       └── split_v1.json         # 데이터 분할 매핑
└── processed/
    ├── windows_balanced_v1.parquet    # 균형 조정된 윈도우
    ├── features_100w_v1.parquet       # 100W feature
    ├── features_200w_v1.parquet       # 200W feature
    └── features_combined_v1.parquet   # 통합 feature
```

---

## 📊 EDA 노트북 실행

```bash
# Jupyter 시작
jupyter notebook

# 브라우저에서 notebooks/eda/ 폴더 열기
# 01_file_inventory_and_splits.ipynb 실행
```

---

## 🔍 주요 산출물 설명

### 1. file_master_v1.parquet
- 전체 파일 메타데이터
- 라벨 정보 (정상/소음/진동/표기없음)
- train/val/test 분할 정보
- **양품 정의**: 100W Sample00, 200W Sample03

### 2. quality_report.csv
- 샘플링 주파수 검증 (510-514 Hz 확인)
- 파일 길이 검증
- 데이터 품질 플래그 (is_usable)

### 3. windows_meta_v1.parquet
- 윈도우 세그먼트 정보 (8초, 50% 중첩)
- **양품**: 시간 기반 분할 (0-60% train, 60-80% val, 80-100% test)
- **불량**: 파일 단위 분할 (70/15/15)

### 4. features_*.parquet
- 시간 영역 feature (RMS, Peak, Crest Factor, Kurtosis, Skewness)
- 채널별 feature (acc-X/Y/Z, Gyro-X/Y/Z, acc-Sum)
- 메타 feature (product, direction)

---

## ⚙️ 설정 커스터마이징

### 윈도우 파라미터 변경

`src/config/params_eda.yaml`:

```yaml
windowing:
  window_sec: 8.0       # 윈도우 길이 (초)
  hop_sec: 4.0          # hop 길이 (초)
  stable_margin: 0.1    # 앞/뒤 제외 비율
  max_windows_per_file: 200  # 파일당 최대 윈도우 수
```

### 클래스 균형 조정

```yaml
balancing:
  target_ratio:
    normal: 1      # 정상
    abnormal: 2    # 불량
  method: "oversample"  # oversample | undersample
```

---

## 🐛 문제 해결

### 엑셀 파일 읽기 오류

```bash
pip install openpyxl --upgrade
```

### 경로 오류

- `src/config/paths.yaml`에서 경로 확인
- 상대 경로 사용 권장

### 메모리 부족

- `max_windows_per_file` 값 줄이기 (예: 100)
- 제품별로 개별 실행

---

## 📈 다음 단계

1. **EDA 노트북 탐색**
   - `01_file_inventory_and_splits.ipynb`: 데이터 분포 확인
   - 추가 노트북 작성 예정 (시간/주파수 영역 분석)

2. **Feature 검증**
   - Feature 분포 확인
   - 클래스 분리 가능성 평가

3. **모델링 (2차 단계)**
   - XGBoost 베이스라인 학습
   - 성능 평가 및 개선

---

## 📞 도움말

- 프로젝트 구조: [README.md](README.md)
- 상세 문서:
  - [00_NeuroM_RoboticArm-진동분석프로젝트개요.md](00_NeuroM_RoboticArm-진동분석프로젝트개요.md)
  - [01_단계별분석전략.md](01_단계별분석전략.md)
  - [02_분석목적-데이터현황-배경지식.md](02_분석목적-데이터현황-배경지식.md)

---

**Last Updated**: 2025-11-17
**Version**: 1.0
