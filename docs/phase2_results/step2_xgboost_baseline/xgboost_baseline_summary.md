# Step 2: XGBoost 베이스라인 - 요약 보고서

**날짜**: 2025-11-17
**특성**: 9개 (RMS 핵심 + Peak + Crest + Kurtosis)
**모델**: XGBoost (최대 반복 1000회)

---

## 1. 특성 집합

```python
feature_set = [
    'acc_Y_rms',
    'acc_X_rms',
    'Gyro_Y_rms',
    'Gyro_X_rms',
    'acc_Y_peak',
    'acc_Sum_peak',
    'acc_Y_crest',
    'acc_Y_kurtosis',
    'acc_Sum_kurtosis',
]
```

---

## 2. 교차 검증 결과

| 지표 | 평균 | 표준편차 |
|--------|------|-----|
| AUC | 0.9768 | 0.0182 |
| Abnormal Recall | 0.913 | 0.032 |
| Abnormal Precision | 0.964 | 0.021 |
| Abnormal F1 | 0.938 | 0.022 |

---

## 3. Test 세트 성능

| 지표 | 값 |
|--------|-------|
| **AUC** | **0.7084** |
| Abnormal Recall | 0.691 |
| Abnormal Precision | 0.931 |
| Abnormal F1 | 0.793 |

---

## 4. Step 1 베이스라인과 비교

| 지표 | Step 1 (RMS 규칙) | Step 2 (XGBoost) | 개선도 |
|--------|-------------------|------------------|-------------|
| Test AUC | 0.6960 | 0.7084 | +0.0124 |
| Abnormal Recall | 0.464 | 0.691 | +0.227 |
| Abnormal Precision | 0.978 | 0.931 | -0.047 |

---

## 5. 상위 5개 중요 특성

| 순위 | 특성 | 중요도 |
|------|---------|------------|
| 1 | acc_Y_peak | 29.1 |
| 2 | acc_Sum_peak | 27.2 |
| 3 | acc_Y_kurtosis | 26.2 |
| 4 | acc_X_rms | 23.5 |
| 5 | acc_Y_rms | 19.6 |

---

## 6. 주요 발견사항

### 목표 달성도
- CV 평균 AUC > 0.75: ✅ (0.9768)
- Test AUC > 0.70: ✅ (0.7084)
- Recall > 0.80: ❌ (0.691)

### 장점
- Step 1 베이스라인 대비 AUC 개선
- 다중 특성이 복잡한 패턴 포착
- 특성 중요도가 해석 가능성 제공

### 다음 단계
- Step 2-2: XGBoost + RMS 규칙 하이브리드 테스트
- Step 2-3: 특성 중요도 상세 분석
- Step 3: 밴드 RMS 특성 추가 (1-10, 10-50, 50-150 Hz)

**생성된 파일**:
- `xgboost_baseline_v1.json`: 학습된 모델
- `cv_results.csv`: 교차 검증 결과
- `eval_results.csv`: Val/Test 평가
- `feature_importance.csv`: 특성 중요도
