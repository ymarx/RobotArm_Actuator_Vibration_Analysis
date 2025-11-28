# ECMiner 레이블 시스템 통합 테스트 보고서

## 1. 개요

ECMiner 패키지에 유연한 레이블 매핑 시스템을 통합하여, "시험전 시료 표기내용.xlsx" 파일 기반의 레이블링을 지원합니다.

**구현 날짜**: 2025-11-25
**테스트 상태**: ✅ 성공

## 2. 구현 내용

### 2.1 새로운 파일
1. **ecminer_labels.csv**: 시료별 레이블 정보 (29개 시료)
2. **ecminer_config.yaml**: 레이블 매핑 전략 설정 파일
   - `default`: 소음을 약한 양품(weight=0.5)으로 포함
   - `strict`: 정상만 양품
   - `inclusive`: 정상+소음을 동등하게 양품 처리

### 2.2 수정된 파일
1. **ecminer_stage1_py_node.py**: 레이블 시스템 통합
   - `load_label_file()`: CSV/Excel 레이블 파일 로드
   - `load_config()`: YAML 설정 로드
   - `apply_label_strategy()`: 레이블 전략 적용 (overrides + weights)
   - `process_file()`: label_raw, label_binary, label_weight 파라미터 추가

2. **requirements.txt**: pyyaml>=6.0 의존성 추가

## 3. 레이블 전략 (default)

### 3.1 매핑 규칙
```yaml
PASS (양품):
  - 정상 (weight: 1.0)
  - 소음 (weight: 0.5)  # weak positive

FAIL (불량):
  - 진동 (weight: 1.0)
  - 표기없음 (weight: 1.0)

Special Overrides:
  - 200W Sample03: 진동 → 정상 (실제 측정 결과 양품 확인)
```

### 3.2 레이블 분포 (전체 661 윈도우)
```
정상:     19 윈도우 (label_binary=1, weight=1.0)
소음:    114 윈도우 (label_binary=1, weight=0.5)
진동:     95 윈도우 (label_binary=0, weight=1.0)
         22 윈도우 (label_binary=1, weight=1.0) # 200W Sample03만
표기없음: 411 윈도우 (label_binary=0, weight=1.0)
```

## 4. 테스트 결과

### 4.1 데이터 처리
```
✅ 입력 파일: 60개 CSV 파일
✅ 레이블 파일: 29개 레이블
✅ 총 윈도우: 661개
✅ Train/Val/Test: 485/84/92
✅ 중복 window_id: 없음
✅ NaN 값: 없음
```

### 4.2 레이블 시스템 검증
```
✅ label_raw 컬럼 생성됨 (정상/소음/진동/표기없음)
✅ label_binary 올바르게 매핑됨 (1/0)
✅ label_weight 올바르게 적용됨 (소음=0.5, 나머지=1.0)
✅ Special override 작동 확인 (200W Sample03: 진동→정상)
```

### 4.3 XGBoost 성능 테스트
```
파라미터:
  - max_depth: 3
  - learning_rate: 0.1
  - n_estimators: 100
  - sample_weight: label_weight 적용

결과:
  - Test AUC: 0.9429
  - Accuracy: 0.96
  - Precision (불량): 0.99
  - Recall (양품): 0.86
```

## 5. 출력 형식

### 5.1 메타데이터 컬럼
```csv
window_id,file_id,dataset_type,label_raw,label_binary,label_weight,product,sample,direction
```

### 5.2 특징 컬럼 (18개)
```
기본 통계 (9개):
  - acc_Y_rms, acc_Y_peak, acc_Y_crest
  - acc_Sum_rms, acc_Sum_peak, acc_Sum_crest
  - Gyro_Y_rms, Gyro_Y_peak, Gyro_Y_crest

밴드 RMS (9개):
  - acc_Y_rms_low/mid/high
  - acc_Sum_rms_low/mid/high
  - Gyro_Y_rms_low/mid/high
```

## 6. 향후 확장 옵션

### 6.1 RMS 기반 재분류 (현재 비활성)
```yaml
rms_reclassification:
  enabled: false  # 향후 활성화 가능
  feature: "acc_Y_rms"
  threshold_method: "normal_range"  # mean ± 2*std
  reclassify_to: "정상"
```

**근거**: 표기없음의 80.5%가 정상 RMS 범위 내
- 정상 평균: 0.125 ± 0.036
- 표기없음 평균: 0.173 ± 0.105

### 6.2 다른 전략 사용
```bash
# ecminer_config.yaml 수정
label_strategy: "strict"  # 또는 "inclusive"
```

## 7. 사용 방법

### 7.1 ECMiner에서 실행
```
1. ECMiner GUI에서 Python Node로 ecminer_stage1_py_node.py 등록
2. 입력 CSV 형식: file_path, file_id, label (옵션)
3. 출력: label_raw, label_binary, label_weight 포함된 특징 데이터
```

### 7.2 로컬 테스트
```bash
cd ECMiner_Package
python ecminer_stage1_py_node.py
# 출력: ../ecminer_output_labeled.csv
```

## 8. 품질 보증

### 8.1 검증 항목
- [x] 레이블 파일 정상 로드
- [x] 설정 파일 정상 파싱
- [x] 레이블 매핑 정확성
- [x] Special override 작동
- [x] 가중치 적용 확인
- [x] 중복 window_id 없음
- [x] NaN 값 없음
- [x] XGBoost 학습 성공
- [x] 성능 지표 양호 (AUC > 0.9)

### 8.2 호환성
- ✅ ECMiner 입력 형식 호환
- ✅ 기존 Phase1-3 분석 코드와 호환
- ✅ Python 3.9+ 지원
- ✅ 의존성 최소화 (pyyaml 추가만)

## 9. 결론

ECMiner 패키지에 레이블 시스템이 성공적으로 통합되었습니다.

**핵심 성과**:
1. "시험전 시료 표기내용.xlsx" 기반 레이블링 지원
2. 유연한 전략 시스템 (default/strict/inclusive)
3. Special override 지원 (200W Sample03)
4. 레이블 가중치 시스템 (소음=0.5)
5. 향후 RMS 기반 재분류 옵션 준비

**테스트 상태**: 모든 기능 정상 작동 확인 ✅
