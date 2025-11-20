"""
Step 3-4: XGBoost v2 with Band RMS Features
============================================
Goal: Re-train XGBoost with expanded feature set including Band RMS
      to improve Recall while maintaining Precision at threshold=0.5

Feature Set v2 (18 features):
- Original core features (9): RMS core, Peak, Crest, Kurtosis
- New Band RMS features (9): acc_Y/acc_Sum/Gyro_Y × low/mid/high

Expected Improvements:
- Test Recall: 0.691 → ~0.75+ (at threshold=0.5)
- Maintain Normal Recall: ~0.60+
- Reduce overfitting gap if possible

Comparison Targets:
- Step 2: XGBoost baseline (9 features, threshold=0.5)
- Step 3-2: Hybrid Rule (XGBoost + RMS rule)
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
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_DIR = PROJECT_ROOT / "docs" / "phase2_results" / "step3_4_xgboost_v2_band_rms"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("="*80)
print("STEP 3-4: XGBoost v2 with Band RMS Features")
print("="*80)

# ============================================================================
# 1. Load Data (v2 with Band RMS)
# ============================================================================
print("\n[1] Loading enhanced features...")

features = pd.read_parquet(DATA_DIR / 'features_combined_v2_with_band_rms.parquet')
features_clean = features.dropna()

print(f"Total windows: {len(features)}")
print(f"After removing NaN: {len(features_clean)}")

train_df = features_clean[features_clean['split_set'] == 'train'].copy()
val_df = features_clean[features_clean['split_set'] == 'val'].copy()
test_df = features_clean[features_clean['split_set'] == 'test'].copy()

print(f"Train: {len(train_df)} (Normal: {(train_df['label_binary']==1).sum()}, Abnormal: {(train_df['label_binary']==0).sum()})")
print(f"Val:   {len(val_df)} (Normal: {(val_df['label_binary']==1).sum()}, Abnormal: {(val_df['label_binary']==0).sum()})")
print(f"Test:  {len(test_df)} (Normal: {(test_df['label_binary']==1).sum()}, Abnormal: {(test_df['label_binary']==0).sum()})")

# ============================================================================
# 2. Define Feature Set v2
# ============================================================================
print("\n[2] Defining feature set v2...")

# Original core features (9 from Step 2)
original_features = [
    # RMS core (4)
    "acc_Y_rms", "acc_X_rms", "Gyro_Y_rms", "Gyro_X_rms",
    # Peak (2)
    "acc_Y_peak", "acc_Sum_peak",
    # Crest factor (1)
    "acc_Y_crest",
    # Kurtosis (2)
    "acc_Y_kurtosis", "acc_Sum_kurtosis",
]

# New Band RMS features (9)
band_rms_features = [
    "acc_Y_rms_low", "acc_Y_rms_mid", "acc_Y_rms_high",
    "acc_Sum_rms_low", "acc_Sum_rms_mid", "acc_Sum_rms_high",
    "Gyro_Y_rms_low", "Gyro_Y_rms_mid", "Gyro_Y_rms_high",
]

# Combined feature set v2
feature_set_v2 = original_features + band_rms_features

print(f"Feature Set v2: {len(feature_set_v2)} features")
print(f"  Original: {len(original_features)}")
print(f"  Band RMS: {len(band_rms_features)}")

# Verify all features exist
missing_features = [f for f in feature_set_v2 if f not in features_clean.columns]
if missing_features:
    print(f"❌ Missing features: {missing_features}")
    raise ValueError("Some features are missing from the dataset")
else:
    print(f"✓ All features present")

# ============================================================================
# 3. Prepare Data
# ============================================================================
print("\n[3] Preparing data...")

X_train = train_df[feature_set_v2].values
y_train = train_df['label_binary'].values  # 1=Normal, 0=Abnormal
X_val = val_df[feature_set_v2].values
y_val = val_df['label_binary'].values
X_test = test_df[feature_set_v2].values
y_test = test_df['label_binary'].values

# Class counts
normal_count = (y_train == 1).sum()
abnormal_count = (y_train == 0).sum()
scale_pos_weight = abnormal_count / normal_count

print(f"Class balance (Train):")
print(f"  Normal: {normal_count}, Abnormal: {abnormal_count}")
print(f"  scale_pos_weight: {scale_pos_weight:.2f}")

# ============================================================================
# 4. Train XGBoost v2 with Cross-Validation
# ============================================================================
print("\n[4] Training XGBoost v2 with 5-Fold CV...")

params = {
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'scale_pos_weight': scale_pos_weight,
    'max_depth': 4,
    'learning_rate': 0.01,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': 42,
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

cv_results = []
fold_models = []

for fold, (train_idx, val_idx) in enumerate(cv.split(X_train, y_train), 1):
    X_fold_train = X_train[train_idx]
    y_fold_train = y_train[train_idx]
    X_fold_val = X_train[val_idx]
    y_fold_val = y_train[val_idx]

    dtrain = xgb.DMatrix(X_fold_train, label=y_fold_train, feature_names=feature_set_v2)
    dval = xgb.DMatrix(X_fold_val, label=y_fold_val, feature_names=feature_set_v2)

    # Train
    evals = [(dtrain, 'train'), (dval, 'val')]
    bst = xgb.train(
        params,
        dtrain,
        num_boost_round=1000,
        evals=evals,
        early_stopping_rounds=50,
        verbose_eval=False
    )

    fold_models.append(bst)

    # Predict on fold validation
    y_pred_proba = bst.predict(dval)
    y_pred = (y_pred_proba >= 0.5).astype(int)

    # Metrics
    cm = confusion_matrix(y_fold_val, y_pred)
    tn, fp, fn, tp = cm.ravel()

    # Abnormal metrics
    precision_abn = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    recall_abn = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    f1_abn = 2 * precision_abn * recall_abn / (precision_abn + recall_abn) if (precision_abn + recall_abn) > 0 else 0.0

    # Normal metrics
    precision_norm = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall_norm = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1_norm = 2 * precision_norm * recall_norm / (precision_norm + recall_norm) if (precision_norm + recall_norm) > 0 else 0.0

    auc_score = roc_auc_score(y_fold_val, y_pred_proba)

    cv_results.append({
        'fold': fold,
        'auc': auc_score,
        'precision_abnormal': precision_abn,
        'recall_abnormal': recall_abn,
        'f1_abnormal': f1_abn,
        'precision_normal': precision_norm,
        'recall_normal': recall_norm,
        'f1_normal': f1_norm,
    })

    print(f"  Fold {fold}: AUC={auc_score:.4f}, Recall(Abn)={recall_abn:.3f}, Precision(Abn)={precision_abn:.3f}")

cv_results_df = pd.DataFrame(cv_results)
cv_results_df.to_csv(OUTPUT_DIR / "cv_results_v2.csv", index=False)

# CV Summary
cv_summary = {
    'mean_auc': cv_results_df['auc'].mean(),
    'std_auc': cv_results_df['auc'].std(),
    'mean_recall_abnormal': cv_results_df['recall_abnormal'].mean(),
    'std_recall_abnormal': cv_results_df['recall_abnormal'].std(),
    'mean_precision_abnormal': cv_results_df['precision_abnormal'].mean(),
    'std_precision_abnormal': cv_results_df['precision_abnormal'].std(),
}

print(f"\nCV Summary:")
print(f"  Mean AUC: {cv_summary['mean_auc']:.4f} ± {cv_summary['std_auc']:.4f}")
print(f"  Mean Recall (Abnormal): {cv_summary['mean_recall_abnormal']:.3f} ± {cv_summary['std_recall_abnormal']:.3f}")
print(f"  Mean Precision (Abnormal): {cv_summary['mean_precision_abnormal']:.3f} ± {cv_summary['std_precision_abnormal']:.3f}")

# ============================================================================
# 5. Train Final Model on Full Training Set
# ============================================================================
print("\n[5] Training final model on full training set...")

dtrain_full = xgb.DMatrix(X_train, label=y_train, feature_names=feature_set_v2)
dval_full = xgb.DMatrix(X_val, label=y_val, feature_names=feature_set_v2)

evals_full = [(dtrain_full, 'train'), (dval_full, 'val')]
bst_final = xgb.train(
    params,
    dtrain_full,
    num_boost_round=1000,
    evals=evals_full,
    early_stopping_rounds=50,
    verbose_eval=False
)

# Save final model
model_path = OUTPUT_DIR / "xgboost_v2_final_model.json"
bst_final.save_model(str(model_path))
print(f"Saved final model to: {model_path}")

# ============================================================================
# 6. Feature Importance
# ============================================================================
print("\n[6] Analyzing feature importance...")

importance = bst_final.get_score(importance_type='gain')
importance_df = pd.DataFrame([
    {'feature': k, 'importance': v}
    for k, v in importance.items()
]).sort_values('importance', ascending=False)

importance_df.to_csv(OUTPUT_DIR / "feature_importance_v2.csv", index=False)

print("\nTop 10 Features by Importance:")
print(importance_df.head(10).to_string(index=False))

# ============================================================================
# 7. Evaluate on Test Set
# ============================================================================
print("\n[7] Evaluating on Test set...")

dtest = xgb.DMatrix(X_test, label=y_test, feature_names=feature_set_v2)
y_test_proba = bst_final.predict(dtest)
y_test_pred = (y_test_proba >= 0.5).astype(int)

# Confusion matrix
cm_test = confusion_matrix(y_test, y_test_pred)
tn, fp, fn, tp = cm_test.ravel()

# Abnormal metrics
precision_abn_test = tn / (tn + fn) if (tn + fn) > 0 else 0.0
recall_abn_test = tn / (tn + fp) if (tn + fp) > 0 else 0.0
f1_abn_test = 2 * precision_abn_test * recall_abn_test / (precision_abn_test + recall_abn_test) if (precision_abn_test + recall_abn_test) > 0 else 0.0

# Normal metrics
precision_norm_test = tp / (tp + fp) if (tp + fp) > 0 else 0.0
recall_norm_test = tp / (tp + fn) if (tp + fn) > 0 else 0.0
f1_norm_test = 2 * precision_norm_test * recall_norm_test / (precision_norm_test + recall_norm_test) if (precision_norm_test + recall_norm_test) > 0 else 0.0

auc_test = roc_auc_score(y_test, y_test_proba)

print(f"\nTest Set Performance (threshold=0.5):")
print(f"  AUC: {auc_test:.4f}")
print(f"\n  Abnormal Class:")
print(f"    Precision: {precision_abn_test:.3f}")
print(f"    Recall:    {recall_abn_test:.3f}")
print(f"    F1:        {f1_abn_test:.3f}")
print(f"\n  Normal Class:")
print(f"    Precision: {precision_norm_test:.3f}")
print(f"    Recall:    {recall_norm_test:.3f}")
print(f"    F1:        {f1_norm_test:.3f}")

# ============================================================================
# 8. Comparison with Baseline
# ============================================================================
print("\n[8] Comparing with baseline models...")

# Load Step 2 baseline results
step2_dir = PROJECT_ROOT / "docs" / "phase2_results" / "step2_xgboost_baseline"
step2_eval = pd.read_csv(step2_dir / "eval_results.csv")

# Load Step 3-2 Hybrid results
step3_2_dir = PROJECT_ROOT / "docs" / "phase2_results" / "step3_2_hybrid_rule"
step3_2_comparison = pd.read_csv(step3_2_dir / "strategy_comparison_test.csv")

# Extract Step 2 Test metrics
step2_test = step2_eval[step2_eval['dataset'] == 'test'].iloc[0]

# Extract Step 3-2 XGBoost and Hybrid metrics
step3_2_xgb = step3_2_comparison[step3_2_comparison['Strategy'] == 'XGBoost (τ=0.5)'].iloc[0]
step3_2_hybrid = step3_2_comparison[step3_2_comparison['Strategy'] == 'Hybrid (XGB | RMS)'].iloc[0]

# Create comparison table
comparison = pd.DataFrame([
    {
        'Model': 'Step 2: XGBoost v1 (9 features)',
        'Features': '9 (original)',
        'AUC': step2_test['auc'],
        'Precision_Abnormal': step2_test['abnormal_precision'],
        'Recall_Abnormal': step2_test['abnormal_recall'],
        'F1_Abnormal': step2_test['abnormal_f1'],
        'Precision_Normal': step2_test['normal_precision'],
        'Recall_Normal': step2_test['normal_recall'],
        'F1_Normal': step2_test['normal_f1'],
    },
    {
        'Model': 'Step 3-2: Hybrid (XGB v1 + RMS rule)',
        'Features': '9 + rule',
        'AUC': np.nan,  # Hybrid doesn't have single AUC
        'Precision_Abnormal': step3_2_hybrid['Precision_Abnormal'],
        'Recall_Abnormal': step3_2_hybrid['Recall_Abnormal'],
        'F1_Abnormal': step3_2_hybrid['F1_Abnormal'],
        'Precision_Normal': step3_2_hybrid['Precision_Normal'],
        'Recall_Normal': step3_2_hybrid['Recall_Normal'],
        'F1_Normal': step3_2_hybrid['F1_Normal'],
    },
    {
        'Model': 'Step 3-4: XGBoost v2 (18 features)',
        'Features': '18 (original + Band RMS)',
        'AUC': auc_test,
        'Precision_Abnormal': precision_abn_test,
        'Recall_Abnormal': recall_abn_test,
        'F1_Abnormal': f1_abn_test,
        'Precision_Normal': precision_norm_test,
        'Recall_Normal': recall_norm_test,
        'F1_Normal': f1_norm_test,
    }
])

comparison.to_csv(OUTPUT_DIR / "model_comparison.csv", index=False)

print("\n" + "="*80)
print("MODEL COMPARISON - TEST SET")
print("="*80)
print(comparison.to_string(index=False))

# Calculate improvements
recall_improvement_vs_v1 = recall_abn_test - step2_test['abnormal_recall']
recall_improvement_vs_hybrid = recall_abn_test - step3_2_hybrid['Recall_Abnormal']

print(f"\nImprovement Analysis:")
print(f"  Recall (Abnormal) vs Step 2:   {recall_improvement_vs_v1:+.3f} ({recall_improvement_vs_v1/step2_test['abnormal_recall']*100:+.1f}%)")
print(f"  Recall (Abnormal) vs Hybrid:   {recall_improvement_vs_hybrid:+.3f} ({recall_improvement_vs_hybrid/step3_2_hybrid['Recall_Abnormal']*100:+.1f}%)")

# ============================================================================
# 9. Generate Summary Report
# ============================================================================
print("\n[9] Generating summary report...")

summary = f"""# Step 3-4: XGBoost v2 with Band RMS Features - Summary

## Goal
Re-train XGBoost with expanded feature set (18 features = 9 original + 9 Band RMS)
to improve Recall while maintaining Precision at threshold=0.5.

## Feature Set v2 (18 features)

### Original Features (9)
- RMS core: acc_Y_rms, acc_X_rms, Gyro_Y_rms, Gyro_X_rms
- Peak: acc_Y_peak, acc_Sum_peak
- Crest: acc_Y_crest
- Kurtosis: acc_Y_kurtosis, acc_Sum_kurtosis

### New Band RMS Features (9)
- acc_Y: low/mid/high (1-10Hz, 10-50Hz, 50-150Hz)
- acc_Sum: low/mid/high
- Gyro_Y: low/mid/high

## Training Configuration
- **Algorithm**: XGBoost with binary:logistic
- **Cross-Validation**: 5-Fold StratifiedKFold
- **scale_pos_weight**: {scale_pos_weight:.2f} (Abnormal/Normal ratio)
- **Hyperparameters**: max_depth=4, learning_rate=0.01, subsample=0.8
- **Early Stopping**: 50 rounds on validation AUC

## Results

### Cross-Validation (Train Set)
- **Mean AUC**: {cv_summary['mean_auc']:.4f} ± {cv_summary['std_auc']:.4f}
- **Mean Recall (Abnormal)**: {cv_summary['mean_recall_abnormal']:.3f} ± {cv_summary['std_recall_abnormal']:.3f}
- **Mean Precision (Abnormal)**: {cv_summary['mean_precision_abnormal']:.3f} ± {cv_summary['std_precision_abnormal']:.3f}

### Test Set Performance (threshold=0.5)
- **AUC**: {auc_test:.4f}
- **Abnormal Class**:
  - Precision: {precision_abn_test:.3f}
  - Recall: {recall_abn_test:.3f}
  - F1: {f1_abn_test:.3f}
- **Normal Class**:
  - Precision: {precision_norm_test:.3f}
  - Recall: {recall_norm_test:.3f}
  - F1: {f1_norm_test:.3f}

## Comparison with Baseline Models

| Model | Features | AUC | Precision (Abn) | Recall (Abn) | F1 (Abn) | Recall (Norm) |
|-------|----------|-----|-----------------|--------------|----------|---------------|
| XGBoost v1 (Step 2) | 9 | {step2_test['auc']:.3f} | {step2_test['abnormal_precision']:.3f} | {step2_test['abnormal_recall']:.3f} | {step2_test['abnormal_f1']:.3f} | {step2_test['normal_recall']:.3f} |
| Hybrid (Step 3-2) | 9 + rule | - | {step3_2_hybrid['Precision_Abnormal']:.3f} | {step3_2_hybrid['Recall_Abnormal']:.3f} | {step3_2_hybrid['F1_Abnormal']:.3f} | {step3_2_hybrid['Recall_Normal']:.3f} |
| **XGBoost v2 (Step 3-4)** | **18** | **{auc_test:.3f}** | **{precision_abn_test:.3f}** | **{recall_abn_test:.3f}** | **{f1_abn_test:.3f}** | **{recall_norm_test:.3f}** |

### Improvements
- **vs XGBoost v1**: Recall (Abn) {recall_improvement_vs_v1:+.3f} ({recall_improvement_vs_v1/step2_test['abnormal_recall']*100:+.1f}%)
- **vs Hybrid**: Recall (Abn) {recall_improvement_vs_hybrid:+.3f} ({recall_improvement_vs_hybrid/step3_2_hybrid['Recall_Abnormal']*100:+.1f}%)

## Top 5 Most Important Features
{importance_df.head(5).to_markdown(index=False)}

## Key Findings

1. **Band RMS Impact**: {'Positive - improved Recall' if recall_improvement_vs_v1 > 0.05 else 'Modest - slight improvement' if recall_improvement_vs_v1 > 0 else 'Negative - decreased Recall'}

2. **Feature Importance**:
   - Top feature: {importance_df.iloc[0]['feature']} ({importance_df.iloc[0]['importance']:.1f})
   - Band RMS features in top 10: {len([f for f in importance_df.head(10)['feature'] if any(band in f for band in ['_low', '_mid', '_high'])])}

3. **Overfitting Check**:
   - CV AUC: {cv_summary['mean_auc']:.4f}
   - Test AUC: {auc_test:.4f}
   - Gap: {cv_summary['mean_auc'] - auc_test:.4f}

4. **Model Choice Recommendation**:
   {'XGBoost v2 recommended - better balance' if recall_abn_test > step3_2_hybrid['Recall_Abnormal'] and recall_norm_test >= 0.5 else 'Hybrid still superior for Recall priority' if step3_2_hybrid['Recall_Abnormal'] > recall_abn_test else 'Further tuning needed'}

## Files Generated
- `xgboost_v2_final_model.json`: Trained model
- `cv_results_v2.csv`: Cross-validation detailed results
- `feature_importance_v2.csv`: Feature importance rankings
- `model_comparison.csv`: Comparison with baseline models
- `step3_4_summary.md`: This summary report

## Next Steps
1. **If XGBoost v2 shows improvement**:
   - Apply Hybrid Rule v2 (XGBoost v2 + Band RMS rules)
   - Test product-specific thresholds (100W vs 200W)

2. **If improvement is modest**:
   - Investigate feature selection (focus on high-importance Band RMS)
   - Consider threshold tuning for v2 model
   - Prepare for Autoencoder/anomaly detection approach

3. **Production Deployment**:
   - Document final model choice (v2 vs Hybrid)
   - Create inference pipeline with feature extraction
   - Plan monitoring and retraining strategy
"""

with open(OUTPUT_DIR / "step3_4_summary.md", 'w') as f:
    f.write(summary)

print(f"Summary saved to: {OUTPUT_DIR / 'step3_4_summary.md'}")

print("\n" + "="*80)
print("STEP 3-4 COMPLETE")
print("="*80)
print(f"\nKey Results:")
print(f"  Test AUC: {auc_test:.4f}")
print(f"  Test Recall (Abnormal): {recall_abn_test:.3f}")
print(f"  Improvement vs Baseline: {recall_improvement_vs_v1:+.3f}")
