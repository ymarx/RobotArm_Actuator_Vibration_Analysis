# 독립 실행 가이드

ECMiner_Package 폴더만으로 완전히 독립적으로 실행하는 방법

## ✅ 검증 완료

**날짜**: 2025-11-25
**상태**: 독립 실행 성공

## 📦 패키지 구성

### 필수 파일

```
ECMiner_Package/
├── ecminer_stage1_py_node.py      ← 메인 스크립트
├── ecminer_labels.csv               ← 레이블 정보 (29개)
├── ecminer_config.yaml              ← 레이블 전략 설정
├── ecminer_input_example.csv        ← 샘플 입력 파일
├── extract_labels_from_excel.py     ← 레이블 추출 스크립트
├── requirements.txt                 ← Python 의존성
├── README_KR.md                     ← 사용 설명서
├── setup_windows.bat                ← Windows 설치 스크립트
└── data/                            ← 원본 데이터
    ├── 100W/
    │   └── *.csv (30개 파일)
    └── 200W/
        └── *.csv (31개 파일)
```

## 🚀 독립 실행 방법

### Step 1: Python 환경 준비

```bash
# Python 3.8+ 확인
python --version

# 의존성 설치
cd ECMiner_Package
pip install -r requirements.txt
```

### Step 2: 스크립트 실행

```bash
# 메인 스크립트 실행
python ecminer_stage1_py_node.py
```

### Step 3: 결과 확인

```
입력 CSV 로드 완료: ecminer_input_example.csv (6개 파일)
레이블 파일 로드 완료: 29개 레이블
레이블 전략: default - 소음을 양품으로 포함하되 약한 가중치 부여

처리 완료:
  - 총 윈도우 수: 61
  - Train: 46
  - Val: 8
  - Test: 7
  - 정상: 61
  - 불량: 0
  - 특징 개수: 18개
  ✓ 중복 window_id 없음

테스트 출력 저장: ../ecminer_output_labeled.csv
```

## 📁 파일 경로 자동 해석

스크립트는 다음 순서로 파일을 자동으로 찾습니다:

### 1. 입력 CSV

1. `ECMiner_Package/ecminer_input_example.csv` (1순위)
2. `부모폴더/ecminer_input_full.csv` (2순위)
3. `ECMiner_Package/ecminer_input_full.csv` (3순위)

### 2. Raw 데이터 파일

1. `ECMiner_Package/data/` (1순위)
2. `부모폴더/data/` (2순위)

**장점**: 
- ECMiner_Package 폴더만 있어도 작동
- 부모 폴더에 data가 있어도 작동
- 두 환경 모두 지원

## 🔧 커스터마이징

### 입력 파일 변경

다른 입력 파일을 사용하려면:

```bash
# 1. 기존 ecminer_input_example.csv를 복사
cp ecminer_input_example.csv my_input.csv

# 2. my_input.csv 편집 (파일 경로 수정)
# file_path,file_id
# data/100W/...,100W_Sample00_CW_...
# data/200W/...,200W_Sample03_CW_...

# 3. 파일명을 ecminer_input_example.csv로 변경하거나
# 4. 코드에서 input_csv_candidates 수정
```

### 레이블 전략 변경

```bash
# ecminer_config.yaml 편집
label_strategy: "strict"  # default, strict, inclusive 중 선택
```

### 레이블 업데이트

```bash
# 1. Excel 파일 준비
# ECMiner_Package 폴더에 "시험전 시료 표기내용.xlsx" 배치

# 2. 레이블 추출
python extract_labels_from_excel.py

# 3. 재실행
python ecminer_stage1_py_node.py
```

## ⚠️ 문제 해결

### 오류 1: 입력 CSV를 찾을 수 없음

```
오류: 입력 CSV 파일을 찾을 수 없습니다.

다음 위치 중 하나에 입력 CSV 파일을 배치하세요:
1. ECMiner_Package/ecminer_input_example.csv
2. 부모폴더/ecminer_input_full.csv
```

**해결**: `ecminer_input_example.csv` 파일이 있는지 확인

### 오류 2: Raw 데이터 파일을 찾을 수 없음

```
경고: 파일 100W_Sample00_CW... 처리 중 오류 발생: No such file or directory
```

**해결**: 
1. `data/100W/`, `data/200W/` 폴더가 있는지 확인
2. 입력 CSV의 `file_path` 경로가 정확한지 확인

### 오류 3: 레이블 파일 없음

```
FileNotFoundError: 레이블 파일을 찾을 수 없습니다: ecminer_labels.csv
```

**해결**: 
```bash
python extract_labels_from_excel.py  # 레이블 CSV 재생성
```

## 📊 검증 테스트 결과

### 테스트 환경 1: ECMiner_Package 단독

```
위치: /tmp/ecminer_standalone_test/ECMiner_Package/
상황: 부모 폴더에 아무것도 없음

✅ 입력 CSV: ecminer_input_example.csv 로드 성공
✅ 레이블 파일: ecminer_labels.csv 로드 성공
✅ Raw 데이터: ECMiner_Package/data/ 에서 로드 성공
✅ 출력 생성: ecminer_output_labeled.csv (61 윈도우)
✅ 레이블 컬럼: label_raw, label_binary, label_weight 포함
```

### 테스트 환경 2: 부모 폴더에 data 있음

```
위치: [프로젝트루트]/ECMiner_Package/
상황: 부모 폴더에 data/ 폴더 있음

✅ 입력 CSV: ecminer_input_example.csv 로드 성공
✅ 레이블 파일: ecminer_labels.csv 로드 성공
✅ Raw 데이터: 자동으로 적절한 위치에서 로드 성공
✅ 출력 생성: 정상
```

### 파일 경로 자동 탐색 검증

| 파일 위치 | ECMiner_Package/data/ | 부모폴더/data/ | 결과 |
|----------|----------------------|---------------|------|
| 100W CSV | ✓ | ✗ | ECMiner_Package/data/에서 로드 |
| 200W CSV | ✗ | ✓ | 부모폴더/data/에서 로드 |
| 100W+200W | ✓ | ✓ | ECMiner_Package/data/ 우선 |
| 없음 | ✗ | ✗ | 명확한 오류 메시지 |

## 🎯 결론

**ECMiner_Package 폴더만으로 완전히 독립적으로 실행 가능합니다!**

### 동료에게 전달 시

1. **ECMiner_Package 폴더 전체**를 압축하여 전달
2. 압축 해제 후 `pip install -r requirements.txt`
3. `python ecminer_stage1_py_node.py` 실행
4. 결과 확인: `ecminer_output_labeled.csv`

### 포함 데이터

- **샘플 데이터**: 6개 파일 (100W 4개, 200W 2개)
- **전체 데이터**: 61개 파일 (100W 30개, 200W 31개)
- **레이블**: 29개 시료 정보
- **출력**: label_raw, label_binary, label_weight + 18개 특징

**검증 상태**: ✅ 완전 독립 실행 성공
