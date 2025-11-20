# Step 3-4: 밴드 RMS 특성을 포함한 XGBoost v2 - 요약

## 목표
확장된 특성 집합(18개 특성 = 기존 9개 + 밴드 RMS 9개)으로 XGBoost를 재학습하여
임계값=0.5에서 Precision을 유지하면서 Recall을 개선합니다.

## 특성 집합 v2 (18개 특성)

### 기존 특성 (9개)
- RMS 핵심: acc_Y_rms, acc_X_rms, Gyro_Y_rms, Gyro_X_rms
- Peak: acc_Y_peak, acc_Sum_peak
- Crest: acc_Y_crest
- Kurtosis: acc_Y_kurtosis, acc_Sum_kurtosis

### 새로운 밴드 RMS 특성 (9개)
- acc_Y: low/mid/high (1-10Hz, 10-50Hz, 50-150Hz)
- acc_Sum: low/mid/high
- Gyro_Y: low/mid/high

## 학습 구성
- **알고리즘**: XGBoost (binary:logistic)
- **교차 검증**: 5-Fold StratifiedKFold
- **scale_pos_weight**: 0.88 (Abnormal/Normal 비율)
- **하이퍼파라미터**: max_depth=4, learning_rate=0.01, subsample=0.8
- **조기 중단**: validation AUC 기준 50 라운드

## 결과

### 교차 검증 (Train 세트)
- **평균 AUC**: 0.9968 ± 0.0027
- **평균 Recall (Abnormal)**: 0.941 ± 0.021
- **평균 Precision (Abnormal)**: 0.984 ± 0.011

### Test 세트 성능 (임계값=0.5)
- **AUC**: 0.8115
- **Abnormal 클래스**:
  - Precision: 0.948
  - Recall: 0.567
  - F1: 0.710
- **Normal 클래스**:
  - Precision: 0.208
  - Recall: 0.786
  - F1: 0.328

## 베이스라인 모델과 비교

| 모델 | 특성 수 | AUC | Precision (Abn) | Recall (Abn) | F1 (Abn) | Recall (Norm) |
|-------|----------|-----|-----------------|--------------|----------|---------------|
| XGBoost v1 (Step 2) | 9 | 0.708 | 0.931 | 0.691 | 0.793 | 0.643 |
| Hybrid (Step 3-2) | 9 + 규칙 | - | 0.929 | 0.804 | 0.862 | 0.571 |
| **XGBoost v2 (Step 3-4)** | **18** | **0.811** | **0.948** | **0.567** | **0.710** | **0.786** |

### 개선도
- **vs XGBoost v1**: Recall (Abn) -0.124 (-17.9%)
- **vs Hybrid**: Recall (Abn) -0.237 (-29.5%)

## 상위 5개 중요 특성
| feature         |   importance |
|:----------------|-------------:|
| acc_Sum_rms_mid |      33.8815 |
| Gyro_X_rms      |      33.1771 |
| acc_Y_kurtosis  |      32.5807 |
| Gyro_Y_rms_low  |      31.585  |
| acc_Y_peak      |      27.9972 |

## 주요 발견사항

1. **밴드 RMS 영향**: 부정적 - Recall 감소

2. **특성 중요도**:
   - 최고 특성: acc_Sum_rms_mid (33.9)
   - 상위 10위 내 밴드 RMS 특성: 5개

3. **과적합 확인**:
   - CV AUC: 0.9968
   - Test AUC: 0.8115
   - 차이: 0.1853

4. **모델 선택 권장사항**:
   Recall 우선순위에서는 여전히 Hybrid가 우수

## 생성된 파일
- `xgboost_v2_final_model.json`: 학습된 모델
- `cv_results_v2.csv`: 교차 검증 상세 결과
- `feature_importance_v2.csv`: 특성 중요도 순위
- `model_comparison.csv`: 베이스라인 모델과 비교
- `step3_4_summary.md`: 본 요약 보고서

## 다음 단계
1. **XGBoost v2가 개선을 보이는 경우**:
   - 하이브리드 규칙 v2 적용 (XGBoost v2 + 밴드 RMS 규칙)
   - 제품별 임계값 테스트 (100W vs 200W)

2. **개선이 미미한 경우**:
   - 특성 선택 조사 (중요도 높은 밴드 RMS에 집중)
   - v2 모델의 임계값 튜닝 고려
   - Autoencoder/이상 감지 접근법 준비

3. **프로덕션 배포**:
   - 최종 모델 선택 문서화 (v2 vs Hybrid)
   - 특성 추출을 포함한 추론 파이프라인 생성
   - 모니터링 및 재학습 전략 계획
