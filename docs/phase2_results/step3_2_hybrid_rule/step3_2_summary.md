# Step 3-2: 하이브리드 규칙 요약

## 목표
XGBoost 예측과 RMS 임계값 규칙을 결합하여 Recall을 개선하면서
Precision 트레이드오프를 평가합니다.

## 구성
- **XGBoost 임계값**: 0.5 (연구 베이스라인, 0.625 아님)
- **RMS 임계값**: 0.15 (acc_Y_rms, Step 1에서)
- **하이브리드 전략**: `pred_hybrid = pred_xgb | pred_rms`
  - 둘 중 하나라도 Abnormal 예측 시 Abnormal 판정

## 임계값=0.5 사용 근거
임계값=0.5 (0.625가 아닌) 사용 이유:
1. 하이브리드의 보완 효과를 더 명확하게 평가
2. Normal/Abnormal 감지의 균형 유지
3. 향후 비교를 위한 공정한 연구 베이스라인으로 사용

Step 3-1의 임계값=0.625는 "초보수 모드" 옵션으로 보류됩니다.

## 결과 - 전략 비교 (Test 세트)

| 전략 | Precision (Abn) | Recall (Abn) | F1 (Abn) | Precision (Norm) | Recall (Norm) | F1 (Norm) |
|----------|-----------------|--------------|----------|------------------|---------------|-----------|
| XGBoost (τ=0.5) | 0.931 | 0.691 | 0.793 | 0.231 | 0.643 | 0.340 |
| RMS (T=0.15) | 0.978 | 0.464 | 0.629 | 0.200 | 0.929 | 0.329 |
| Hybrid (XGB \| RMS) | 0.929 | 0.804 | 0.862 | 0.296 | 0.571 | 0.390 |

### 주요 지표 변화 (XGBoost → Hybrid)
- **Recall (Abnormal)**: 0.691 → 0.804 (+0.113, +16.4%)
- **Precision (Abnormal)**: 0.931 → 0.929 (-0.002, -0.2%)

## 하이브리드 개선 분석

### ✅ 추가된 True Positive
하이브리드가 XGBoost가 놓친 **11개의 추가 True Abnormal 케이스**를 포착했습니다.

**개선된 케이스의 특성:**
- acc_Y_rms: 평균=0.1883, 중앙값=0.1915
- xgb_prob: 평균=0.5544, 중앙값=0.5464
- 해석: XGBoost는 이 케이스들에 Normal 확률 (prob >= 0.5)을 부여했지만, RMS 규칙이 포착 (acc_Y_rms > 0.15)

### ⚠️ 추가된 False Positive
하이브리드가 **1개의 False Positive**를 추가했습니다 (Normal 샘플을 Abnormal로 잘못 판정).

**추가된 FP 케이스의 특성:**
- acc_Y_rms: 평균=0.1569, 중앙값=0.1569
- xgb_prob: 평균=0.5462, 중앙값=0.5462
- 해석: XGBoost는 이를 Normal로 올바르게 분류했지만, RMS 규칙이 플래그 (acc_Y_rms > 0.15)

## 주요 발견사항

1. **하이브리드 효과성**: 하이브리드가 성공적으로 Recall 개선
2. **Precision 트레이드오프**: 허용 가능한 정밀도 손실
3. **보완 가치**: RMS 규칙이 XGBoost가 놓친 높은 진동 케이스를 포착하여 보완

## 생성된 파일
- `strategy_comparison_test.csv`: 세 가지 전략 모두의 성능 지표
- `hybrid_improved_cases.csv`: 하이브리드가 XGBoost 누락을 포착한 케이스
- `hybrid_added_fp_cases.csv`: 하이브리드가 추가한 False Positive
- `hybrid_decision_space.png`: 결정 경계 시각화
- `hybrid_scatter_data.csv`: 산점도 데이터 (xgb_prob vs acc_Y_rms)
- `step3_2_summary.md`: 본 요약 보고서

## 다음 단계
- Step 3-3으로 진행: 밴드 RMS 특성 추가 (1-10Hz, 10-50Hz, 50-150Hz)
- 확장된 특성 집합으로 XGBoost 재학습
- 더 나은 특성이 극단적 임계값이나 하이브리드 규칙의 필요성을 줄이는지 평가
