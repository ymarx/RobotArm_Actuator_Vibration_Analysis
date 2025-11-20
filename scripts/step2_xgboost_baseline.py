"""
Step 2: XGBoost Baseline Model
시간영역 features (9개)로 XGBoost baseline 구축

Feature Set (Small, 9개):
- RMS core (4개): acc_Y_rms, acc_X_rms, Gyro_Y_rms, Gyro_X_rms
- Peak (2개): acc_Y_peak, acc_Sum_peak
- Crest factor (1개): acc_Y_crest
- Kurtosis (2개): acc_Y_kurtosis, acc_Sum_kurtosis

목표:
1. CV Mean AUC > 0.75
2. Test AUC > 0.70 (Step 1의 0.696 개선)
3. Recall > 0.80 (Step 1의 0.464 개선)
4. Feature importance 분석
"""

import pandas as pd
import numpy as np
from pathlib import Path
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    roc_auc_score, precision_recall_fscore_support,
    confusion_matrix, roc_curve, auc
)
import json

# Paths
BASE_PATH = Path(__file__).parent.parent
DATA_PATH = BASE_PATH / "data" / "processed"
OUTPUT_PATH = BASE_PATH / "docs" / "phase2_results" / "step2_xgboost_baseline"
OUTPUT_PATH.mkdir(exist_ok=True, parents=True)

print("=" * 80)
print("Step 2: XGBoost Baseline Model")
print("=" * 80)
print()

# ============================================================================
# 1. Load Data
# ============================================================================
print("[1/7] Loading data...")
features = pd.read_parquet(DATA_PATH / 'features_combined_v1.parquet')
features_clean = features.dropna()

print(f"  Total windows: {len(features)}")
print(f"  After removing NaN: {len(features_clean)}")
print()

# ============================================================================
# 2. Define Feature Set
# ============================================================================
print("[2/7] Defining feature set...")

feature_set = [
    # RMS core (4개)
    "acc_Y_rms", "acc_X_rms", "Gyro_Y_rms", "Gyro_X_rms",
    # Peak (2개)
    "acc_Y_peak", "acc_Sum_peak",
    # Crest factor (1개)
    "acc_Y_crest",
    # Kurtosis (2개)
    "acc_Y_kurtosis", "acc_Sum_kurtosis",
]

print(f"  Feature count: {len(feature_set)}")
print(f"  Features: {feature_set}")
print()

# Verify all features exist
missing_features = [f for f in feature_set if f not in features_clean.columns]
if missing_features:
    print(f"  ❌ Missing features: {missing_features}")
    raise ValueError("Some features are missing from the dataset")
else:
    print(f"  ✓ All features present in dataset")
print()

# ============================================================================
# 3. Prepare Train/Val/Test Splits
# ============================================================================
print("[3/7] Preparing splits...")

train_df = features_clean[features_clean['split_set'] == 'train'].copy()
val_df = features_clean[features_clean['split_set'] == 'val'].copy()
test_df = features_clean[features_clean['split_set'] == 'test'].copy()

X_train = train_df[feature_set].values
y_train = train_df['label_binary'].values  # 1=Normal, 0=Abnormal
X_val = val_df[feature_set].values
y_val = val_df['label_binary'].values
X_test = test_df[feature_set].values
y_test = test_df['label_binary'].values

print(f"  Train: {len(X_train)} (Normal: {y_train.sum()}, Abnormal: {len(y_train)-y_train.sum()})")
print(f"  Val:   {len(X_val)} (Normal: {y_val.sum()}, Abnormal: {len(y_val)-y_val.sum()})")
print(f"  Test:  {len(X_test)} (Normal: {y_test.sum()}, Abnormal: {len(y_test)-y_test.sum()})")
print()

# ============================================================================
# 4. XGBoost Parameters
# ============================================================================
print("[4/7] Setting XGBoost parameters...")

# Calculate scale_pos_weight (Abnormal/Normal ratio)
abnormal_count = len(y_train) - y_train.sum()
normal_count = y_train.sum()
scale_pos_weight = abnormal_count / normal_count

params = {
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'scale_pos_weight': scale_pos_weight,
    'max_depth': 4,  # 작게 시작 (과적합 방지)
    'learning_rate': 0.01,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': 42,
    'n_estimators': 1000,
}

print(f"  Parameters:")
for k, v in params.items():
    print(f"    {k}: {v}")
print()

# ============================================================================
# 5. 5-Fold Cross-Validation
# ============================================================================
print("[5/7] Running 5-Fold Cross-Validation...")

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_results = []
cv_models = []

for fold, (train_idx, val_idx) in enumerate(cv.split(X_train, y_train), 1):
    print(f"\n  Fold {fold}/5")
    print(f"  " + "-" * 40)

    X_tr, X_vl = X_train[train_idx], X_train[val_idx]
    y_tr, y_vl = y_train[train_idx], y_train[val_idx]

    # Create DMatrix
    dtrain = xgb.DMatrix(X_tr, label=y_tr)
    dval = xgb.DMatrix(X_vl, label=y_vl)

    # Train
    evals = [(dtrain, 'train'), (dval, 'val')]
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=params['n_estimators'],
        evals=evals,
        early_stopping_rounds=50,
        verbose_eval=False
    )

    # Predict
    y_pred_proba = model.predict(dval)
    y_pred = (y_pred_proba > 0.5).astype(int)

    # Metrics
    auc_score = roc_auc_score(y_vl, y_pred_proba)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_vl, y_pred, average='binary', pos_label=1  # Normal as positive
    )

    # For Abnormal detection (invert)
    y_vl_abnormal = 1 - y_vl
    y_pred_abnormal = 1 - y_pred
    precision_abn, recall_abn, f1_abn, _ = precision_recall_fscore_support(
        y_vl_abnormal, y_pred_abnormal, average='binary', pos_label=1
    )

    cv_results.append({
        'fold': fold,
        'auc': auc_score,
        'normal_precision': precision,
        'normal_recall': recall,
        'normal_f1': f1,
        'abnormal_precision': precision_abn,
        'abnormal_recall': recall_abn,
        'abnormal_f1': f1_abn,
        'best_iteration': model.best_iteration,
    })

    cv_models.append(model)

    print(f"    AUC: {auc_score:.4f}")
    print(f"    Normal    - P: {precision:.3f}, R: {recall:.3f}, F1: {f1:.3f}")
    print(f"    Abnormal  - P: {precision_abn:.3f}, R: {recall_abn:.3f}, F1: {f1_abn:.3f}")

cv_results_df = pd.DataFrame(cv_results)
cv_results_df.to_csv(OUTPUT_PATH / 'cv_results.csv', index=False)

print()
print("  " + "=" * 40)
print(f"  CV Results Summary:")
print(f"  " + "=" * 40)
print(f"    Mean AUC: {cv_results_df['auc'].mean():.4f} ± {cv_results_df['auc'].std():.4f}")
print(f"    Mean Abnormal Recall: {cv_results_df['abnormal_recall'].mean():.3f} ± {cv_results_df['abnormal_recall'].std():.3f}")
print(f"    Mean Abnormal Precision: {cv_results_df['abnormal_precision'].mean():.3f} ± {cv_results_df['abnormal_precision'].std():.3f}")
print(f"    Mean Abnormal F1: {cv_results_df['abnormal_f1'].mean():.3f} ± {cv_results_df['abnormal_f1'].std():.3f}")
print()

# ============================================================================
# 6. Train Final Model on Full Train Set
# ============================================================================
print("[6/7] Training final model on full train set...")

dtrain_full = xgb.DMatrix(X_train, label=y_train)
dval_full = xgb.DMatrix(X_val, label=y_val)

evals_full = [(dtrain_full, 'train'), (dval_full, 'val')]
final_model = xgb.train(
    params,
    dtrain_full,
    num_boost_round=params['n_estimators'],
    evals=evals_full,
    early_stopping_rounds=50,
    verbose_eval=False
)

print(f"  ✓ Best iteration: {final_model.best_iteration}")
print()

# Save model
final_model.save_model(str(OUTPUT_PATH / 'xgboost_baseline_v1.json'))
print(f"  ✓ Saved: xgboost_baseline_v1.json")
print()

# ============================================================================
# 7. Evaluate on Validation and Test Sets
# ============================================================================
print("[7/7] Evaluating on Validation and Test sets...")

def evaluate_model(model, X, y, dataset_name):
    """Evaluate model on a dataset"""
    dmatrix = xgb.DMatrix(X)
    y_pred_proba = model.predict(dmatrix)
    y_pred = (y_pred_proba > 0.5).astype(int)

    # Metrics (Normal as positive)
    auc_score = roc_auc_score(y, y_pred_proba)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y, y_pred, average='binary', pos_label=1
    )

    # Abnormal metrics
    y_abnormal = 1 - y
    y_pred_abnormal = 1 - y_pred
    precision_abn, recall_abn, f1_abn, _ = precision_recall_fscore_support(
        y_abnormal, y_pred_abnormal, average='binary', pos_label=1
    )

    # Confusion matrix
    cm = confusion_matrix(y, y_pred)
    tn, fp, fn, tp = cm.ravel()

    return {
        'dataset': dataset_name,
        'auc': auc_score,
        'normal_precision': precision,
        'normal_recall': recall,
        'normal_f1': f1,
        'abnormal_precision': precision_abn,
        'abnormal_recall': recall_abn,
        'abnormal_f1': f1_abn,
        'tn': tn,  # True Abnormal
        'fp': fp,  # False Abnormal
        'fn': fn,  # False Normal
        'tp': tp,  # True Normal
    }

# Evaluate
val_eval = evaluate_model(final_model, X_val, y_val, 'val')
test_eval = evaluate_model(final_model, X_test, y_test, 'test')

eval_results_df = pd.DataFrame([val_eval, test_eval])
eval_results_df.to_csv(OUTPUT_PATH / 'eval_results.csv', index=False)

print()
print("  Validation Set:")
print(f"    AUC: {val_eval['auc']:.4f}")
print(f"    Normal    - P: {val_eval['normal_precision']:.3f}, R: {val_eval['normal_recall']:.3f}, F1: {val_eval['normal_f1']:.3f}")
print(f"    Abnormal  - P: {val_eval['abnormal_precision']:.3f}, R: {val_eval['abnormal_recall']:.3f}, F1: {val_eval['abnormal_f1']:.3f}")

print()
print("  Test Set:")
print(f"    AUC: {test_eval['auc']:.4f}")
print(f"    Normal    - P: {test_eval['normal_precision']:.3f}, R: {test_eval['normal_recall']:.3f}, F1: {test_eval['normal_f1']:.3f}")
print(f"    Abnormal  - P: {test_eval['abnormal_precision']:.3f}, R: {test_eval['abnormal_recall']:.3f}, F1: {test_eval['abnormal_f1']:.3f}")
print()

# ============================================================================
# 8. Feature Importance
# ============================================================================
print("Analyzing feature importance...")

importance_dict = final_model.get_score(importance_type='gain')
importance_df = pd.DataFrame([
    {'feature': k, 'importance': v}
    for k, v in importance_dict.items()
]).sort_values('importance', ascending=False)

# Map feature indices to names
feature_map = {f'f{i}': name for i, name in enumerate(feature_set)}
importance_df['feature_name'] = importance_df['feature'].map(feature_map)

importance_df.to_csv(OUTPUT_PATH / 'feature_importance.csv', index=False)
print(f"  ✓ Saved: feature_importance.csv")
print()

print("  Top 5 Important Features:")
for idx, row in importance_df.head(5).iterrows():
    print(f"    {row['feature_name']}: {row['importance']:.1f}")
print()

# ============================================================================
# 9. Compare with Step 1 Baseline
# ============================================================================
print("=" * 80)
print("Performance Comparison: XGBoost vs Step 1 RMS Rule")
print("=" * 80)
print()

# Step 1 baseline (from previous results)
step1_test_auc = 0.696
step1_test_recall = 0.464
step1_test_precision = 0.978

print(f"{'Metric':<25} {'Step 1 (RMS Rule)':<20} {'Step 2 (XGBoost)':<20} {'Improvement':<15}")
print("-" * 80)
print(f"{'Test AUC':<25} {step1_test_auc:<20.4f} {test_eval['auc']:<20.4f} {test_eval['auc']-step1_test_auc:+.4f}")
print(f"{'Abnormal Recall':<25} {step1_test_recall:<20.3f} {test_eval['abnormal_recall']:<20.3f} {test_eval['abnormal_recall']-step1_test_recall:+.3f}")
print(f"{'Abnormal Precision':<25} {step1_test_precision:<20.3f} {test_eval['abnormal_precision']:<20.3f} {test_eval['abnormal_precision']-step1_test_precision:+.3f}")
print()

# ============================================================================
# 10. Summary Report
# ============================================================================
print("Generating summary report...")

summary_lines = []
summary_lines.append("# Step 2: XGBoost Baseline - Summary Report")
summary_lines.append("")
summary_lines.append("**Date**: 2025-11-17")
summary_lines.append(f"**Features**: {len(feature_set)} (RMS core + Peak + Crest + Kurtosis)")
summary_lines.append(f"**Model**: XGBoost with {params['n_estimators']} max iterations")
summary_lines.append("")
summary_lines.append("---")
summary_lines.append("")
summary_lines.append("## 1. Feature Set")
summary_lines.append("")
summary_lines.append("```python")
summary_lines.append("feature_set = [")
for f in feature_set:
    summary_lines.append(f"    '{f}',")
summary_lines.append("]")
summary_lines.append("```")
summary_lines.append("")
summary_lines.append("---")
summary_lines.append("")
summary_lines.append("## 2. Cross-Validation Results")
summary_lines.append("")
summary_lines.append("| Metric | Mean | Std |")
summary_lines.append("|--------|------|-----|")
summary_lines.append(f"| AUC | {cv_results_df['auc'].mean():.4f} | {cv_results_df['auc'].std():.4f} |")
summary_lines.append(f"| Abnormal Recall | {cv_results_df['abnormal_recall'].mean():.3f} | {cv_results_df['abnormal_recall'].std():.3f} |")
summary_lines.append(f"| Abnormal Precision | {cv_results_df['abnormal_precision'].mean():.3f} | {cv_results_df['abnormal_precision'].std():.3f} |")
summary_lines.append(f"| Abnormal F1 | {cv_results_df['abnormal_f1'].mean():.3f} | {cv_results_df['abnormal_f1'].std():.3f} |")
summary_lines.append("")
summary_lines.append("---")
summary_lines.append("")
summary_lines.append("## 3. Test Set Performance")
summary_lines.append("")
summary_lines.append("| Metric | Value |")
summary_lines.append("|--------|-------|")
summary_lines.append(f"| **AUC** | **{test_eval['auc']:.4f}** |")
summary_lines.append(f"| Abnormal Recall | {test_eval['abnormal_recall']:.3f} |")
summary_lines.append(f"| Abnormal Precision | {test_eval['abnormal_precision']:.3f} |")
summary_lines.append(f"| Abnormal F1 | {test_eval['abnormal_f1']:.3f} |")
summary_lines.append("")
summary_lines.append("---")
summary_lines.append("")
summary_lines.append("## 4. Comparison with Step 1 Baseline")
summary_lines.append("")
summary_lines.append("| Metric | Step 1 (RMS Rule) | Step 2 (XGBoost) | Improvement |")
summary_lines.append("|--------|-------------------|------------------|-------------|")
summary_lines.append(f"| Test AUC | {step1_test_auc:.4f} | {test_eval['auc']:.4f} | {test_eval['auc']-step1_test_auc:+.4f} |")
summary_lines.append(f"| Abnormal Recall | {step1_test_recall:.3f} | {test_eval['abnormal_recall']:.3f} | {test_eval['abnormal_recall']-step1_test_recall:+.3f} |")
summary_lines.append(f"| Abnormal Precision | {step1_test_precision:.3f} | {test_eval['abnormal_precision']:.3f} | {test_eval['abnormal_precision']-step1_test_precision:+.3f} |")
summary_lines.append("")
summary_lines.append("---")
summary_lines.append("")
summary_lines.append("## 5. Top 5 Important Features")
summary_lines.append("")
summary_lines.append("| Rank | Feature | Importance |")
summary_lines.append("|------|---------|------------|")
for rank, (idx, row) in enumerate(importance_df.head(5).iterrows(), 1):
    summary_lines.append(f"| {rank} | {row['feature_name']} | {row['importance']:.1f} |")
summary_lines.append("")
summary_lines.append("---")
summary_lines.append("")
summary_lines.append("## 6. Key Findings")
summary_lines.append("")

# Goal achievement
cv_auc_goal = cv_results_df['auc'].mean() > 0.75
test_auc_goal = test_eval['auc'] > 0.70
recall_goal = test_eval['abnormal_recall'] > 0.80

summary_lines.append("### Goal Achievement")
summary_lines.append(f"- CV Mean AUC > 0.75: {'✅' if cv_auc_goal else '❌'} ({cv_results_df['auc'].mean():.4f})")
summary_lines.append(f"- Test AUC > 0.70: {'✅' if test_auc_goal else '❌'} ({test_eval['auc']:.4f})")
summary_lines.append(f"- Recall > 0.80: {'✅' if recall_goal else '❌'} ({test_eval['abnormal_recall']:.3f})")
summary_lines.append("")
summary_lines.append("### Strengths")
summary_lines.append("- Improved AUC over Step 1 baseline")
summary_lines.append("- Multiple features capture complex patterns")
summary_lines.append("- Feature importance provides interpretability")
summary_lines.append("")
summary_lines.append("### Next Steps")
summary_lines.append("- Step 2-2: Test XGBoost + RMS Rule hybrid")
summary_lines.append("- Step 2-3: Analyze feature importance in detail")
summary_lines.append("- Step 3: Add band RMS features (1-10, 10-50, 50-150 Hz)")
summary_lines.append("")
summary_lines.append("**Files Generated**:")
summary_lines.append("- `xgboost_baseline_v1.json`: Trained model")
summary_lines.append("- `cv_results.csv`: Cross-validation results")
summary_lines.append("- `eval_results.csv`: Val/Test evaluation")
summary_lines.append("- `feature_importance.csv`: Feature importance")

with open(OUTPUT_PATH / 'xgboost_baseline_summary.md', 'w', encoding='utf-8') as f:
    f.write('\n'.join(summary_lines))

print(f"  ✓ Saved: xgboost_baseline_summary.md")
print()

# ============================================================================
# Final Summary
# ============================================================================
print("=" * 80)
print("Step 2-1 Complete!")
print("=" * 80)
print()
print(f"Output directory: {OUTPUT_PATH}")
print()
print("Files generated:")
print("  1. xgboost_baseline_v1.json - Trained model")
print("  2. cv_results.csv - Cross-validation results")
print("  3. eval_results.csv - Validation/Test evaluation")
print("  4. feature_importance.csv - Feature importance")
print("  5. xgboost_baseline_summary.md - Summary report")
print()
print("=" * 80)
print("Goal Achievement:")
print("=" * 80)
print(f"  CV Mean AUC > 0.75: {'✅ PASS' if cv_auc_goal else '❌ FAIL'} ({cv_results_df['auc'].mean():.4f})")
print(f"  Test AUC > 0.70:    {'✅ PASS' if test_auc_goal else '❌ FAIL'} ({test_eval['auc']:.4f})")
print(f"  Recall > 0.80:      {'✅ PASS' if recall_goal else '❌ FAIL'} ({test_eval['abnormal_recall']:.3f})")
print()

if cv_auc_goal and test_auc_goal and recall_goal:
    print("🎉 모든 목표 달성! Step 2-2 (Hybrid Rule)로 진행 가능합니다.")
else:
    print("⚠️  일부 목표 미달. Feature engineering 또는 하이퍼파라미터 튜닝 고려 필요.")
print()
