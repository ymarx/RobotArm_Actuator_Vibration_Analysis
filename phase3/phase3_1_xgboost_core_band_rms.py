"""
Phase 3-1: Refined XGBoost with Core Band RMS Features
========================================================
Step 3-4의 과적합 문제 해결:
1. 핵심 Band RMS 특성만 선택 (3-4개)
2. StratifiedGroupKFold로 CV 누수 방지
3. 강화된 정규화 파라미터 적용

선택 특성:
- 기존 9개 특성 유지
- 추가: acc_Y_rms_high (2.61x), Gyro_Y_rms_high (2.08x), Gyro_Y_rms_low (2.09x)
- 총 12개 특성
"""

import pandas as pd
import numpy as np
from pathlib import Path
import xgboost as xgb
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
import matplotlib.pyplot as plt
import json

# 경로 설정
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "docs" / "phase3_results" / "phase3_1_xgboost_core"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("="*80)
print("PHASE 3-1: XGBoost with Core Band RMS Features")
print("="*80)

# ==============================================================================
# 1. 데이터 로드
# ==============================================================================
print("\n[1] 데이터 로드 중...")

features_df = pd.read_parquet(DATA_DIR / "processed" / "features_combined_v2_with_band_rms.parquet")
features_clean = features_df.dropna()

print(f"총 샘플 수: {len(features_clean)}")
print(f"Train: {len(features_clean[features_clean['split_set']=='train'])}")
print(f"Val:   {len(features_clean[features_clean['split_set']=='val'])}")
print(f"Test:  {len(features_clean[features_clean['split_set']=='test'])}")

# ==============================================================================
# 2. 핵심 Band RMS 특성 선택
# ==============================================================================
print("\n[2] 특성 선택 중...")

# 기존 9개 특성 (Phase 2 Step 3-1에서 사용)
base_features = [
    'acc_Y_rms',       # acc_Y 관련
    'acc_Y_peak',
    'acc_Y_crest',
    'acc_Sum_rms',     # acc_Sum 관련
    'acc_Sum_peak',
    'acc_Sum_crest',
    'Gyro_Y_rms',      # Gyro_Y 관련
    'Gyro_Y_peak',
    'Gyro_Y_crest'
]

# 핵심 Band RMS 특성 3개 (분석에서 가장 높은 비율을 보인 특성)
core_band_rms_features = [
    'acc_Y_rms_high',    # 2.61x ratio (가장 높음)
    'Gyro_Y_rms_high',   # 2.08x ratio
    'Gyro_Y_rms_low'     # 2.09x ratio
]

# 전체 특성 리스트
selected_features = base_features + core_band_rms_features

print(f"\n선택된 특성 ({len(selected_features)}개):")
print("  기존 특성 (9개):", base_features)
print("  추가 Band RMS (3개):", core_band_rms_features)

# ==============================================================================
# 3. Train/Val/Test 분할
# ==============================================================================
print("\n[3] 데이터 분할 중...")

train_df = features_clean[features_clean['split_set'] == 'train'].copy()
val_df = features_clean[features_clean['split_set'] == 'val'].copy()
test_df = features_clean[features_clean['split_set'] == 'test'].copy()

X_train = train_df[selected_features]
y_train = train_df['label_binary']
groups_train = train_df['file_id']  # GroupKFold를 위한 그룹

X_val = val_df[selected_features]
y_val = val_df['label_binary']

X_test = test_df[selected_features]
y_test = test_df['label_binary']

print(f"\nTrain: X={X_train.shape}, y={y_train.shape}")
print(f"  Normal: {(y_train==1).sum()}, Abnormal: {(y_train==0).sum()}")
print(f"Val:   X={X_val.shape}, y={y_val.shape}")
print(f"  Normal: {(y_val==1).sum()}, Abnormal: {(y_val==0).sum()}")
print(f"Test:  X={X_test.shape}, y={y_test.shape}")
print(f"  Normal: {(y_test==1).sum()}, Abnormal: {(y_test==0).sum()}")

# ==============================================================================
# 4. StratifiedGroupKFold CV로 하이퍼파라미터 튜닝
# ==============================================================================
print("\n[4] StratifiedGroupKFold CV로 모델 훈련 중...")

# 하이퍼파라미터 (강화된 정규화)
params = {
    'max_depth': 3,              # 이전: 4 → 감소
    'min_child_weight': 5,       # 이전: 3 → 증가
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'reg_lambda': 5,             # 이전: 1 → 증가
    'reg_alpha': 1,              # 이전: 0 → 추가
    'learning_rate': 0.1,
    'n_estimators': 100,
    'random_state': 42,
    'eval_metric': 'logloss',
    'scale_pos_weight': len(y_train[y_train==0]) / len(y_train[y_train==1])
}

print(f"\n하이퍼파라미터:")
for key, value in params.items():
    print(f"  {key}: {value}")

# StratifiedGroupKFold 설정 (5-fold)
n_splits = 5
sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)

cv_results = []
fold_models = []

print(f"\nStratifiedGroupKFold {n_splits}-fold CV 시작...")

for fold, (train_idx, val_idx) in enumerate(sgkf.split(X_train, y_train, groups_train), 1):
    print(f"\n--- Fold {fold}/{n_splits} ---")

    X_fold_train = X_train.iloc[train_idx]
    y_fold_train = y_train.iloc[train_idx]
    X_fold_val = X_train.iloc[val_idx]
    y_fold_val = y_train.iloc[val_idx]

    print(f"  Fold Train: {X_fold_train.shape}, Normal={sum(y_fold_train==1)}, Abnormal={sum(y_fold_train==0)}")
    print(f"  Fold Val:   {X_fold_val.shape}, Normal={sum(y_fold_val==1)}, Abnormal={sum(y_fold_val==0)}")

    # 모델 훈련
    model = xgb.XGBClassifier(**params)
    model.fit(
        X_fold_train, y_fold_train,
        eval_set=[(X_fold_val, y_fold_val)],
        verbose=False
    )

    # 예측
    y_pred_proba = model.predict_proba(X_fold_val)[:, 1]
    y_pred = model.predict(X_fold_val)

    # 평가
    acc = accuracy_score(y_fold_val, y_pred)
    prec_normal = precision_score(y_fold_val, y_pred, pos_label=1, zero_division=0)
    rec_normal = recall_score(y_fold_val, y_pred, pos_label=1, zero_division=0)
    prec_abnormal = precision_score(y_fold_val, y_pred, pos_label=0, zero_division=0)
    rec_abnormal = recall_score(y_fold_val, y_pred, pos_label=0, zero_division=0)
    auc = roc_auc_score(y_fold_val, y_pred_proba)

    print(f"  Accuracy: {acc:.3f}")
    print(f"  AUC: {auc:.3f}")
    print(f"  Normal - Precision: {prec_normal:.3f}, Recall: {rec_normal:.3f}")
    print(f"  Abnormal - Precision: {prec_abnormal:.3f}, Recall: {rec_abnormal:.3f}")

    cv_results.append({
        'fold': fold,
        'accuracy': acc,
        'auc': auc,
        'precision_normal': prec_normal,
        'recall_normal': rec_normal,
        'precision_abnormal': prec_abnormal,
        'recall_abnormal': rec_abnormal
    })

    fold_models.append(model)

# CV 결과 요약
cv_df = pd.DataFrame(cv_results)
print(f"\n{'='*80}")
print("CV 결과 요약:")
print(f"{'='*80}")
print(cv_df)
print(f"\n평균 성능:")
print(f"  Accuracy: {cv_df['accuracy'].mean():.3f} ± {cv_df['accuracy'].std():.3f}")
print(f"  AUC: {cv_df['auc'].mean():.3f} ± {cv_df['auc'].std():.3f}")
print(f"  Normal Recall: {cv_df['recall_normal'].mean():.3f} ± {cv_df['recall_normal'].std():.3f}")
print(f"  Abnormal Recall: {cv_df['recall_abnormal'].mean():.3f} ± {cv_df['recall_abnormal'].std():.3f}")

cv_df.to_csv(OUTPUT_DIR / "cv_results.csv", index=False)

# ==============================================================================
# 5. 전체 Train 데이터로 최종 모델 훈련
# ==============================================================================
print("\n[5] 전체 Train 데이터로 최종 모델 훈련 중...")

final_model = xgb.XGBClassifier(**params)
final_model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    verbose=False
)

print("최종 모델 훈련 완료")

# ==============================================================================
# 6. Test 세트 평가
# ==============================================================================
print("\n[6] Test 세트 평가 중...")

y_test_pred_proba = final_model.predict_proba(X_test)[:, 1]
y_test_pred = final_model.predict(X_test)

# 성능 지표 계산
test_acc = accuracy_score(y_test, y_test_pred)
test_prec_normal = precision_score(y_test, y_test_pred, pos_label=1, zero_division=0)
test_rec_normal = recall_score(y_test, y_test_pred, pos_label=1, zero_division=0)
test_f1_normal = f1_score(y_test, y_test_pred, pos_label=1, zero_division=0)
test_prec_abnormal = precision_score(y_test, y_test_pred, pos_label=0, zero_division=0)
test_rec_abnormal = recall_score(y_test, y_test_pred, pos_label=0, zero_division=0)
test_f1_abnormal = f1_score(y_test, y_test_pred, pos_label=0, zero_division=0)
test_auc = roc_auc_score(y_test, y_test_pred_proba)

print(f"\n{'='*80}")
print("TEST 세트 성능:")
print(f"{'='*80}")
print(f"Accuracy: {test_acc:.3f}")
print(f"AUC: {test_auc:.3f}")
print(f"\nNormal (Class 1):")
print(f"  Precision: {test_prec_normal:.3f}")
print(f"  Recall: {test_rec_normal:.3f}")
print(f"  F1-Score: {test_f1_normal:.3f}")
print(f"\nAbnormal (Class 0):")
print(f"  Precision: {test_prec_abnormal:.3f}")
print(f"  Recall: {test_rec_abnormal:.3f}")
print(f"  F1-Score: {test_f1_abnormal:.3f}")

# Confusion Matrix
cm = confusion_matrix(y_test, y_test_pred)
print(f"\nConfusion Matrix:")
print(cm)
print(f"\n상세 분류 리포트:")
print(classification_report(y_test, y_test_pred, target_names=['Abnormal', 'Normal']))

# ==============================================================================
# 7. 이전 결과와 비교
# ==============================================================================
print("\n[7] 이전 모델과 비교...")

# Phase 2 Step 3-4 (XGBoost v2 with all Band RMS) 결과 로드
try:
    step3_4_results = pd.read_csv(
        PROJECT_ROOT / "docs" / "phase2_results" / "step3_4_xgboost_v2_band_rms" / "test_results.csv"
    )
    step3_4_acc = step3_4_results['accuracy'].values[0]
    step3_4_auc = step3_4_results['auc'].values[0]
    step3_4_rec_normal = step3_4_results['recall_normal'].values[0]
    step3_4_rec_abnormal = step3_4_results['abnormal_recall'].values[0]
except:
    # 이전 세션 요약에서 가져온 수치
    step3_4_acc = 0.784
    step3_4_auc = 0.811
    step3_4_rec_normal = 0.567
    step3_4_rec_abnormal = 0.804

comparison = pd.DataFrame({
    'Model': ['XGBoost Baseline (Step 2)', 'XGBoost v2 (Step 3-4)', 'XGBoost v3 (Phase 3-1)'],
    'Features': ['9 base features', '18 features (9 base + 9 Band RMS)', '12 features (9 base + 3 core Band RMS)'],
    'CV_Method': ['StratifiedKFold', 'StratifiedKFold', 'StratifiedGroupKFold'],
    'Test_Accuracy': [
        0.784,  # Step 2 baseline (이전 세션에서)
        step3_4_acc,
        test_acc
    ],
    'Test_AUC': [
        0.811,  # Step 2 baseline
        step3_4_auc,
        test_auc
    ],
    'Test_Recall_Normal': [
        0.691,  # Step 2 baseline
        step3_4_rec_normal,
        test_rec_normal
    ],
    'Test_Recall_Abnormal': [
        0.804,  # Step 2 baseline
        step3_4_rec_abnormal,
        test_rec_abnormal
    ],
    'CV_AUC': [
        0.913,  # Step 2에서 보고된 CV AUC
        0.997,  # Step 3-4에서 보고된 CV AUC (과적합)
        cv_df['auc'].mean()
    ]
})

print(f"\n{'='*80}")
print("모델 비교:")
print(f"{'='*80}")
print(comparison.to_string(index=False))

# CV-Test gap 계산
comparison['CV_Test_Gap'] = comparison['CV_AUC'] - comparison['Test_AUC']
print(f"\nCV-Test AUC Gap:")
for idx, row in comparison.iterrows():
    print(f"  {row['Model']}: {row['CV_Test_Gap']:.3f}")

comparison.to_csv(OUTPUT_DIR / "model_comparison.csv", index=False)

# ==============================================================================
# 8. 특성 중요도 분석
# ==============================================================================
print("\n[8] 특성 중요도 분석 중...")

feature_importance = pd.DataFrame({
    'feature': selected_features,
    'importance': final_model.feature_importances_
}).sort_values('importance', ascending=False)

print("\n특성 중요도:")
print(feature_importance.to_string(index=False))

feature_importance.to_csv(OUTPUT_DIR / "feature_importance.csv", index=False)

# 특성 중요도 시각화
plt.figure(figsize=(10, 6))
plt.barh(range(len(feature_importance)), feature_importance['importance'])
plt.yticks(range(len(feature_importance)), feature_importance['feature'])
plt.xlabel('Importance')
plt.title('Feature Importance - XGBoost v3 (Core Band RMS)')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "feature_importance.png", dpi=150, bbox_inches='tight')
plt.close()

print(f"특성 중요도 그래프 저장: {OUTPUT_DIR / 'feature_importance.png'}")

# ==============================================================================
# 9. 결과 저장
# ==============================================================================
print("\n[9] 결과 저장 중...")

# Test 결과 저장
test_results = pd.DataFrame({
    'accuracy': [test_acc],
    'auc': [test_auc],
    'precision_normal': [test_prec_normal],
    'recall_normal': [test_rec_normal],
    'f1_normal': [test_f1_normal],
    'precision_abnormal': [test_prec_abnormal],
    'recall_abnormal': [test_rec_abnormal],
    'f1_abnormal': [test_f1_abnormal]
})

test_results.to_csv(OUTPUT_DIR / "test_results.csv", index=False)

# Confusion Matrix 저장
cm_df = pd.DataFrame(cm,
                     index=['Actual_Abnormal', 'Actual_Normal'],
                     columns=['Pred_Abnormal', 'Pred_Normal'])
cm_df.to_csv(OUTPUT_DIR / "confusion_matrix.csv")

# 모델 저장
final_model.save_model(OUTPUT_DIR / "xgboost_v3_core_band_rms.json")

# 설정 저장
config = {
    'selected_features': selected_features,
    'base_features': base_features,
    'core_band_rms_features': core_band_rms_features,
    'n_features': len(selected_features),
    'hyperparameters': params,
    'cv_method': 'StratifiedGroupKFold',
    'n_splits': n_splits,
    'cv_auc_mean': float(cv_df['auc'].mean()),
    'cv_auc_std': float(cv_df['auc'].std()),
    'test_auc': float(test_auc),
    'test_accuracy': float(test_acc),
    'test_recall_normal': float(test_rec_normal),
    'test_recall_abnormal': float(test_rec_abnormal)
}

with open(OUTPUT_DIR / "config.json", 'w', encoding='utf-8') as f:
    json.dump(config, f, indent=2, ensure_ascii=False)

print(f"\n결과 저장 완료:")
print(f"  - test_results.csv")
print(f"  - cv_results.csv")
print(f"  - model_comparison.csv")
print(f"  - feature_importance.csv")
print(f"  - confusion_matrix.csv")
print(f"  - xgboost_v3_core_band_rms.json")
print(f"  - config.json")

print("\n" + "="*80)
print("PHASE 3-1 완료")
print("="*80)
