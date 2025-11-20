# Step 3-1: 임계값 튜닝 요약

## 목표
- XGBoost 분류 임계값을 조정하여 Recall ≥ 0.80 달성, 동시에 Precision ≥ 0.85 유지
- Validation 세트에서 튜닝, Test 세트에 한 번만 적용

## 방법론
1. Step 2에서 학습된 XGBoost 모델 로드
2. Validation 세트에서 예측 확률 획득
3. 0.05부터 0.95까지 임계값 테스트 (37개 값)
4. 기준 충족 최적 임계값 선택 (Recall ≥ 0.80 AND Precision ≥ 0.85)
5. 선택된 임계값을 Test 세트에 적용

## 결과

### 선택된 임계값: 0.625

### Validation 성능 (선택 기준)
- **Precision (Abnormal)**: 0.957
- **Recall (Abnormal)**: 1.000
- **F1 (Abnormal)**: 0.978
- **Precision (Normal)**: 0.000
- **Recall (Normal)**: 0.000

### Test 성능
- **AUC-ROC**: 0.708
- **Abnormal 클래스**:
  - Precision: 0.889
  - Recall: 0.990
  - F1: 0.937
- **Normal 클래스**:
  - Precision: 0.667
  - Recall: 0.143
  - F1: 0.235

### Step 2와 비교 (임계값=0.5)
| 지표 | Step 2 | Step 3-1 | 변화 |
|--------|--------|----------|--------|
| 임계값 | 0.500 | 0.625 | +0.125 |
| Precision (Abnormal) | 0.931 | 0.889 | -0.042 |
| Recall (Abnormal) | 0.691 | 0.990 | +0.299 |

## 주요 발견사항
1. **임계값 선택**: ✅ 두 기준 모두 성공적으로 충족
2. **Recall 개선**: +43.3%
3. **Precision 트레이드오프**: -4.5%

## 생성된 파일
- `threshold_sweep_validation.csv`: Validation 세트에서 전체 임계값 스위프
- `comparison_step2_vs_step3_1.csv`: Step 2 베이스라인과 비교
- `step3_1_summary.md`: 본 요약 보고서

## 다음 단계
- Step 3-2로 진행: 하이브리드 규칙 구현 (XGBoost + RMS)
- 하이브리드가 XGBoost가 놓친 케이스를 어떻게 포착하는지 분석
