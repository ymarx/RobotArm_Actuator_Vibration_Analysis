# Phase 2 시작 체크리스트

**목표**: XGBoost Baseline 모델 학습 및 평가
**예상 소요**: 2-3 시간
**전제 조건**: Phase 1 EDA 완료 ✅

---

## ✅ Phase 1 완료 확인

- [x] 데이터 품질 기준 조정 (95.0% 사용 가능)
- [x] 데이터 누수 검증 (5개 테스트 통과)
- [x] Feature 추출 (49개 시간 영역)
- [x] EDA 분석 완료
- [x] Phase 2 준비사항 정리

---

## 🎯 Phase 2 목표

### 주요 목표
1. **XGBoost Baseline 성능 확립**: AUC-ROC > 0.80
2. **Feature Importance 파악**: 상위 20개 중요 feature 선정
3. **검증 전략 수립**: 5-Fold CV로 신뢰성 있는 평가

### 성공 기준
- ✅ Train AUC > 0.85
- ✅ CV Mean AUC > 0.75 (±0.05)
- ✅ Test AUC > 0.70
- ✅ Recall (불량 탐지율) > 0.80
- ✅ Precision > 0.85

---

## 📋 실행 단계

### Step 1: 데이터 준비 (10분)

```python
import pandas as pd
import numpy as np
from pathlib import Path

# 데이터 로드
features = pd.read_parquet('data/processed/features_combined_v1.parquet')

# 결측치 제거 (5개 윈도우)
features_clean = features.dropna()
print(f"After removing NaN: {len(features_clean)} windows")

# Train/Val/Test 분리
train_df = features_clean[features_clean['split_set'] == 'train']
val_df = features_clean[features_clean['split_set'] == 'val']
test_df = features_clean[features_clean['split_set'] == 'test']

# Feature columns
sensor_features = [col for col in features_clean.columns
                   if col.startswith(('acc_', 'Gyro_'))]

# X, y 준비
X_train = train_df[sensor_features]
y_train = train_df['label_binary']
X_val = val_df[sensor_features]
y_val = val_df['label_binary']
X_test = test_df[sensor_features]
y_test = test_df['label_binary']

print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
print(f"Normal/Abnormal - Train: {y_train.sum()}/{len(y_train)-y_train.sum()}")
```

**예상 결과**:
```
After removing NaN: 679 windows
Train: 476, Val: 92, Test: 111
Normal/Abnormal - Train: 160/316
```

---

### Step 2: XGBoost Baseline (30분)

```python
import xgboost as xgb
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix

# 파라미터 설정
params = {
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'scale_pos_weight': (len(y_train) - y_train.sum()) / y_train.sum(),  # 불량/정상
    'max_depth': 6,
    'learning_rate': 0.01,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': 42
}

# DMatrix 생성
dtrain = xgb.DMatrix(X_train, label=y_train)
dval = xgb.DMatrix(X_val, label=y_val)

# 학습
evals = [(dtrain, 'train'), (dval, 'val')]
model = xgb.train(
    params,
    dtrain,
    num_boost_round=1000,
    evals=evals,
    early_stopping_rounds=50,
    verbose_eval=50
)

# 예측
y_pred_proba = model.predict(dval)
y_pred = (y_pred_proba > 0.5).astype(int)

# 평가
print(f"Validation AUC: {roc_auc_score(y_val, y_pred_proba):.4f}")
print("\nClassification Report:")
print(classification_report(y_val, y_pred, target_names=['Abnormal', 'Normal']))
```

**체크포인트**: Val AUC > 0.70 달성 확인

---

### Step 3: Cross-Validation (40분)

```python
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

# 5-Fold CV
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = []

for fold, (train_idx, val_idx) in enumerate(cv.split(X_train, y_train)):
    print(f"\n{'='*50}")
    print(f"Fold {fold+1}/5")
    print(f"{'='*50}")

    X_tr, X_vl = X_train.iloc[train_idx], X_train.iloc[val_idx]
    y_tr, y_vl = y_train.iloc[train_idx], y_train.iloc[val_idx]

    dtrain = xgb.DMatrix(X_tr, label=y_tr)
    dval = xgb.DMatrix(X_vl, label=y_vl)

    model = xgb.train(
        params,
        dtrain,
        num_boost_round=1000,
        evals=[(dtrain, 'train'), (dval, 'val')],
        early_stopping_rounds=50,
        verbose_eval=False
    )

    y_pred = model.predict(dval)
    auc = roc_auc_score(y_vl, y_pred)
    cv_scores.append(auc)
    print(f"Fold {fold+1} AUC: {auc:.4f}")

print(f"\n{'='*50}")
print(f"Cross-Validation Results:")
print(f"{'='*50}")
print(f"Mean AUC: {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")
print(f"Scores: {[f'{s:.4f}' for s in cv_scores]}")
```

**체크포인트**: Mean AUC > 0.75 달성 확인

---

### Step 4: Feature Importance (20분)

```python
# Feature importance 추출
importance = model.get_score(importance_type='gain')
importance_df = pd.DataFrame([
    {'feature': k, 'importance': v}
    for k, v in importance.items()
]).sort_values('importance', ascending=False)

# Top 20 features
print("\nTop 20 Important Features:")
print(importance_df.head(20).to_string(index=False))

# 저장
importance_df.to_csv('claudedocs/feature_importance_baseline.csv', index=False)

# 중요도 누적 분포
importance_df['cumsum'] = importance_df['importance'].cumsum() / importance_df['importance'].sum()
top_n_for_80pct = len(importance_df[importance_df['cumsum'] <= 0.8])
print(f"\n80% 설명력을 위한 feature 수: {top_n_for_80pct}")
```

**체크포인트**: 상위 20개 feature가 전체의 80% 이상 설명력 확인

---

### Step 5: Test Set 평가 (20분)

```python
# Test set 예측
dtest = xgb.DMatrix(X_test, label=y_test)
y_test_pred_proba = model.predict(dtest)
y_test_pred = (y_test_pred_proba > 0.5).astype(int)

# 평가
test_auc = roc_auc_score(y_test, y_test_pred_proba)
print(f"\n{'='*50}")
print(f"Test Set Evaluation")
print(f"{'='*50}")
print(f"Test AUC: {test_auc:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_test_pred, target_names=['Abnormal', 'Normal']))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_test_pred))

# 저장
results = {
    'val_auc': roc_auc_score(y_val, y_pred_proba),
    'cv_mean_auc': np.mean(cv_scores),
    'cv_std_auc': np.std(cv_scores),
    'test_auc': test_auc
}
pd.DataFrame([results]).to_csv('claudedocs/baseline_results.csv', index=False)
```

**체크포인트**: Test AUC > 0.70 달성 확인

---

### Step 6: 결과 정리 및 보고서 (30분)

```python
# 종합 성능 요약
summary = f"""
XGBoost Baseline Model Performance
==================================

Data:
- Train: {len(X_train)} windows (Normal: {y_train.sum()}, Abnormal: {len(y_train)-y_train.sum()})
- Val:   {len(X_val)} windows (Normal: {y_val.sum()}, Abnormal: {len(y_val)-y_val.sum()})
- Test:  {len(X_test)} windows (Normal: {y_test.sum()}, Abnormal: {len(y_test)-y_test.sum()})

Model Parameters:
- max_depth: {params['max_depth']}
- learning_rate: {params['learning_rate']}
- scale_pos_weight: {params['scale_pos_weight']:.2f}

Results:
- Validation AUC:  {results['val_auc']:.4f}
- CV Mean AUC:     {results['cv_mean_auc']:.4f} ± {results['cv_std_auc']:.4f}
- Test AUC:        {results['test_auc']:.4f}

Top 5 Features:
{importance_df.head(5).to_string(index=False)}

Next Steps:
1. Feature Engineering (주파수 영역)
2. Hyperparameter Tuning
3. Threshold Optimization
"""

print(summary)
with open('claudedocs/baseline_summary.txt', 'w') as f:
    f.write(summary)
```

---

## 🚨 주의사항

### 데이터 이슈
1. **Val/Test 불균형**: 정상 샘플 매우 부족 (Val: 4개, Test: 14개)
   - → CV 결과를 더 신뢰
   - → Test 결과는 참고용

2. **클래스 불균형**: 불량:정상 = 2:1 (train)
   - → `scale_pos_weight` 설정 필수
   - → Precision/Recall 균형 고려

### 성능 기대치
- **현실적 목표**: Test AUC 0.70-0.80
- **이유**: Large effect size feature 없음 (Cohen's d < 0.8)
- **개선 방향**: 주파수 영역 feature 추가 시 0.80-0.85 기대

### 시간 배분
- 총 예상 시간: 2.5시간
- Step 1-2 (Baseline): 40분
- Step 3 (CV): 40분
- Step 4-5 (Importance + Test): 40분
- Step 6 (정리): 30분

---

## 📁 생성될 파일

```
claudedocs/
├── feature_importance_baseline.csv
├── baseline_results.csv
└── baseline_summary.txt

models/ (optional)
└── xgboost_baseline_v1.json
```

---

## ✅ 완료 후 체크리스트

- [ ] Baseline 모델 학습 완료
- [ ] CV Mean AUC > 0.75 달성
- [ ] Feature importance 분석 완료
- [ ] Test set 평가 완료
- [ ] 결과 보고서 작성 완료
- [ ] 다음 단계 (Feature Engineering) 계획 수립

---

**준비 완료! Phase 2를 시작하시겠습니까?**
