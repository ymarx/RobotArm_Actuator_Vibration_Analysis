# 레이블 추출 가이드

Excel 파일에서 레이블 CSV를 자동 생성하는 방법을 설명합니다.

## 📋 개요

`extract_labels_from_excel.py` 스크립트는 "시험전 시료 표기내용.xlsx" 파일을 읽어서 `ecminer_labels.csv`를 자동으로 생성합니다.

## 🚀 사용 방법

### 1. Excel 파일 준비

Excel 파일을 다음 위치 중 하나에 배치:

```
ECMiner_Package/
├── 시험전 시료 표기내용.xlsx  ← 옵션 1: 같은 폴더
└── extract_labels_from_excel.py

또는

[프로젝트 루트]/
├── 시험전 시료 표기내용.xlsx  ← 옵션 2: 부모 폴더
└── ECMiner_Package/
    └── extract_labels_from_excel.py
```

### 2. 스크립트 실행

```bash
cd ECMiner_Package
python extract_labels_from_excel.py
```

### 3. 결과 확인

```
============================================================
레이블 Excel → CSV 변환 스크립트
============================================================

✓ Excel 파일 발견: 시험전 시료 표기내용.xlsx
✓ 레이블 추출 완료: 29개

레이블 분포:

  100W:
    - 표기없음: 9개
    - 소음: 3개
    - 진동: 2개
    - 정상: 1개

  200W:
    - 표기없음: 10개
    - 진동: 3개
    - 소음: 1개

✓ CSV 저장 완료: ecminer_labels.csv

생성된 CSV 미리보기 (처음 10개):
product  sample label
   100W       0    정상
   100W       1    소음
   100W       2    진동
   ...

============================================================
변환 완료!
============================================================
```

## 📄 Excel 파일 형식

### 지원하는 형식

```
Row 0: 제목 (시험전 시료 표기내용)
Row 1: 빈 행
Row 2: 헤더 (시료번호, 100W, 200W, ...)
Row 3+: 데이터

예시:
| 시료번호 | 100W | 200W |
|---------|------|------|
| 0       | 정상 |      |
| 1       | 소음 | 진동 |
| 2       | 진동 | 소음 |
| ...     | ...  | ...  |
```

### 주요 특징

- **유연한 제품 컬럼**: 100W, 200W 외에 다른 제품도 자동 인식
- **Sample 번호 제한 없음**: 0, 1, 2, ..., 20, 21, ... 모두 지원
- **빈 셀 처리**: NaN 값은 자동으로 제외
- **자동 정렬**: product, sample 순으로 자동 정렬

## 🔄 레이블 업데이트 워크플로우

### 시나리오 1: 새로운 시료 추가

1. Excel 파일에 새 행 추가
2. 변환 스크립트 실행
3. Stage1 스크립트 재실행

```bash
# 1. Excel 편집: 새 시료 추가
# 2. CSV 재생성
python extract_labels_from_excel.py

# 3. 데이터 전처리 재실행
python ecminer_stage1_py_node.py
```

### 시나리오 2: 기존 레이블 수정

1. Excel 파일에서 레이블 변경
2. 변환 스크립트 실행 (기존 CSV 덮어씌움)
3. Stage1 스크립트 재실행

```bash
# 1. Excel 편집: 레이블 변경
# 2. CSV 재생성 (자동 덮어쓰기)
python extract_labels_from_excel.py

# 3. 데이터 전처리 재실행
python ecminer_stage1_py_node.py
```

### 시나리오 3: 새로운 제품 추가

1. Excel에 새 제품 컬럼 추가 (예: 300W)
2. 변환 스크립트 실행 (자동 인식)
3. Stage1 스크립트 재실행

```bash
# 1. Excel 편집: 300W 컬럼 추가
# 2. CSV 재생성 (자동으로 300W 인식)
python extract_labels_from_excel.py

# 3. 데이터 전처리 재실행
python ecminer_stage1_py_node.py
```

## ⚠️ 주의사항

### 필수 요구사항

1. **Excel 파일명**: 정확히 `시험전 시료 표기내용.xlsx`
2. **헤더 행**: "시료번호" 텍스트 포함 필수
3. **제품 컬럼**: "W"가 포함된 컬럼명 (예: 100W, 200W)

### 일반적인 오류

**오류 1: Excel 파일을 찾을 수 없음**
```
FileNotFoundError: 레이블 Excel 파일을 찾을 수 없습니다.
```
→ Excel 파일을 같은 폴더 또는 부모 폴더에 배치

**오류 2: 헤더를 찾을 수 없음**
```
ValueError: Excel 파일에서 '시료번호' 헤더를 찾을 수 없습니다.
```
→ Excel 파일 형식 확인 (Row 2에 "시료번호" 있는지)

**오류 3: 제품 컬럼 없음**
```
ValueError: 제품 컬럼(100W, 200W)을 찾을 수 없습니다.
```
→ 헤더 행에 "W"가 포함된 컬럼 있는지 확인

## 🔧 고급 옵션

### 수동 CSV 편집

자동 생성된 CSV를 수동으로 편집할 수도 있습니다:

```csv
product,sample,label
100W,0,정상
100W,1,소음
200W,3,진동  ← 수정 가능
300W,0,정상  ← 새 제품 추가 가능
```

**주의**: 다시 `extract_labels_from_excel.py`를 실행하면 수동 편집 내용이 사라집니다.

### 병합 시나리오

두 개의 레이블 소스를 병합하려면:

```bash
# 1. Excel에서 기본 레이블 추출
python extract_labels_from_excel.py

# 2. 수동으로 ecminer_labels.csv 편집
#    (추가 시료나 특수 케이스 추가)

# 3. 백업 생성 (선택사항)
cp ecminer_labels.csv ecminer_labels_custom.csv

# 주의: Excel 재추출 시 수동 편집 내용 손실됨
```

## 📊 출력 형식

### ecminer_labels.csv

```csv
product,sample,label
100W,0,정상
100W,1,소음
100W,2,진동
100W,3,진동
100W,4,표기없음
...
200W,1,진동
200W,2,소음
...
```

- **Encoding**: UTF-8 with BOM (Excel 호환)
- **구분자**: 쉼표 (,)
- **정렬**: product → sample 순

## 🔗 다음 단계

레이블 CSV 생성 후:

1. **전략 선택**: `ecminer_config.yaml` 편집 (선택사항)
2. **데이터 전처리**: `python ecminer_stage1_py_node.py`
3. **결과 확인**: `../ecminer_output_labeled.csv`

자세한 내용은 [README_KR.md](README_KR.md)를 참조하세요.
