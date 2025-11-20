"""
Step 3-1: Threshold Tuning for XGBoost Model
==============================================
Goal: Tune classification threshold to achieve Recall ≥ 0.80 while maintaining Precision ≥ 0.85

Key Points:
- Tune on Validation/CV set, NOT Test set
- Test multiple thresholds from 0.1 to 0.9
- Select best threshold meeting criteria (Recall ≥ 0.80 AND Precision ≥ 0.85)
- Apply selected threshold to Test set ONCE only
- Document threshold-Recall/Precision trade-off
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import confusion_matrix, roc_auc_score, classification_report
import pickle
import os
from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODEL_DIR = PROJECT_ROOT / "docs" / "phase2_results" / "step2_xgboost_baseline"
OUTPUT_DIR = PROJECT_ROOT / "docs" / "phase2_results" / "step3_1_threshold_tuning"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Feature set (same as Step 2)
feature_set = [
    "acc_Y_rms", "acc_X_rms", "Gyro_Y_rms", "Gyro_X_rms",  # RMS core
    "acc_Y_peak", "acc_Sum_peak",  # Peak
    "acc_Y_crest",  # Crest factor
    "acc_Y_kurtosis", "acc_Sum_kurtosis",  # Kurtosis
]

print("="*80)
print("STEP 3-1: XGBoost Threshold Tuning")
print("="*80)

# Load data
print("\n[1] Loading data...")
features = pd.read_parquet(DATA_DIR / 'features_combined_v1.parquet')
features_clean = features.dropna()

train_df = features_clean[features_clean['split_set'] == 'train'].copy()
val_df = features_clean[features_clean['split_set'] == 'val'].copy()
test_df = features_clean[features_clean['split_set'] == 'test'].copy()

print(f"Total windows: {len(features)}")
print(f"After removing NaN: {len(features_clean)}")
print(f"Train: {len(train_df)} samples (Normal: {(train_df['label_binary']==1).sum()}, Abnormal: {(train_df['label_binary']==0).sum()})")
print(f"Val:   {len(val_df)} samples (Normal: {(val_df['label_binary']==1).sum()}, Abnormal: {(val_df['label_binary']==0).sum()})")
print(f"Test:  {len(test_df)} samples (Normal: {(test_df['label_binary']==1).sum()}, Abnormal: {(test_df['label_binary']==0).sum()})")

# Prepare data
X_train = train_df[feature_set].values
y_train = train_df['label_binary'].values
X_val = val_df[feature_set].values
y_val = val_df['label_binary'].values
X_test = test_df[feature_set].values
y_test = test_df['label_binary'].values

# Load trained model from Step 2
print("\n[2] Loading trained XGBoost model from Step 2...")
model_path = MODEL_DIR / "xgboost_baseline_v1.json"
if not model_path.exists():
    raise FileNotFoundError(f"Model not found at {model_path}. Please run step2_xgboost_baseline.py first.")

bst = xgb.Booster()
bst.load_model(str(model_path))

print("Model loaded successfully.")

# Get predictions on Validation set
print("\n[3] Getting predictions on Validation set...")
dval = xgb.DMatrix(X_val, feature_names=feature_set)
y_val_proba = bst.predict(dval)

print(f"Prediction probabilities: min={y_val_proba.min():.4f}, max={y_val_proba.max():.4f}, mean={y_val_proba.mean():.4f}")

# Threshold sweep
print("\n[4] Threshold tuning on Validation set...")
thresholds = np.linspace(0.05, 0.95, 37)  # Fine-grained sweep
results = []

for threshold in thresholds:
    # Predict: prob >= threshold -> Normal (1), else -> Abnormal (0)
    y_val_pred = (y_val_proba >= threshold).astype(int)

    # Confusion matrix
    cm = confusion_matrix(y_val, y_val_pred)
    tn, fp, fn, tp = cm.ravel()

    # Metrics for Abnormal class (label=0)
    # True Positive (Abnormal): Predicted=0, Actual=0 -> tn
    # False Negative (Abnormal): Predicted=1, Actual=0 -> fp
    # Precision (Abnormal) = tn / (tn + fn)
    # Recall (Abnormal) = tn / (tn + fp)

    precision_abnormal = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    recall_abnormal = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    f1_abnormal = 2 * precision_abnormal * recall_abnormal / (precision_abnormal + recall_abnormal) if (precision_abnormal + recall_abnormal) > 0 else 0.0

    # Metrics for Normal class (label=1)
    precision_normal = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall_normal = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1_normal = 2 * precision_normal * recall_normal / (precision_normal + recall_normal) if (precision_normal + recall_normal) > 0 else 0.0

    results.append({
        'threshold': threshold,
        'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp,
        'precision_abnormal': precision_abnormal,
        'recall_abnormal': recall_abnormal,
        'f1_abnormal': f1_abnormal,
        'precision_normal': precision_normal,
        'recall_normal': recall_normal,
        'f1_normal': f1_normal,
    })

results_df = pd.DataFrame(results)

# Save threshold sweep results
results_df.to_csv(OUTPUT_DIR / "threshold_sweep_validation.csv", index=False)
print(f"Threshold sweep results saved to {OUTPUT_DIR / 'threshold_sweep_validation.csv'}")

# Find best threshold meeting criteria
print("\n[5] Selecting best threshold...")
print("Criteria: Recall(Abnormal) >= 0.80 AND Precision(Abnormal) >= 0.85")

candidates = results_df[
    (results_df['recall_abnormal'] >= 0.80) &
    (results_df['precision_abnormal'] >= 0.85)
]

if len(candidates) == 0:
    print("\n⚠️  No threshold meets both criteria. Relaxing constraints...")
    # Try relaxed criteria
    candidates = results_df[results_df['recall_abnormal'] >= 0.80]
    if len(candidates) == 0:
        print("⚠️  No threshold achieves Recall >= 0.80. Showing top 5 by Recall:")
        top5 = results_df.nlargest(5, 'recall_abnormal')[['threshold', 'recall_abnormal', 'precision_abnormal', 'f1_abnormal']]
        print(top5.to_string(index=False))
        best_threshold = results_df.loc[results_df['recall_abnormal'].idxmax(), 'threshold']
    else:
        print(f"⚠️  {len(candidates)} thresholds meet Recall >= 0.80 (but not Precision >= 0.85)")
        # Among those, select highest Precision
        best_idx = candidates['precision_abnormal'].idxmax()
        best_threshold = candidates.loc[best_idx, 'threshold']
else:
    print(f"✅ {len(candidates)} thresholds meet both criteria.")
    # Among candidates, select highest F1
    best_idx = candidates['f1_abnormal'].idxmax()
    best_threshold = candidates.loc[best_idx, 'threshold']

best_row = results_df[results_df['threshold'] == best_threshold].iloc[0]

print(f"\n🎯 Selected Threshold: {best_threshold:.3f}")
print(f"   Validation Performance:")
print(f"   - Precision (Abnormal): {best_row['precision_abnormal']:.3f}")
print(f"   - Recall (Abnormal):    {best_row['recall_abnormal']:.3f}")
print(f"   - F1 (Abnormal):        {best_row['f1_abnormal']:.3f}")
print(f"   - Precision (Normal):   {best_row['precision_normal']:.3f}")
print(f"   - Recall (Normal):      {best_row['recall_normal']:.3f}")

# Apply to Test set ONCE
print("\n[6] Applying selected threshold to Test set (ONCE ONLY)...")
dtest = xgb.DMatrix(X_test, feature_names=feature_set)
y_test_proba = bst.predict(dtest)
y_test_pred = (y_test_proba >= best_threshold).astype(int)

# Evaluate on Test set
cm_test = confusion_matrix(y_test, y_test_pred)
tn_test, fp_test, fn_test, tp_test = cm_test.ravel()

precision_abnormal_test = tn_test / (tn_test + fn_test) if (tn_test + fn_test) > 0 else 0.0
recall_abnormal_test = tn_test / (tn_test + fp_test) if (tn_test + fp_test) > 0 else 0.0
f1_abnormal_test = 2 * precision_abnormal_test * recall_abnormal_test / (precision_abnormal_test + recall_abnormal_test) if (precision_abnormal_test + recall_abnormal_test) > 0 else 0.0

precision_normal_test = tp_test / (tp_test + fp_test) if (tp_test + fp_test) > 0 else 0.0
recall_normal_test = tp_test / (tp_test + fn_test) if (tp_test + fn_test) > 0 else 0.0
f1_normal_test = 2 * precision_normal_test * recall_normal_test / (precision_normal_test + recall_normal_test) if (precision_normal_test + recall_normal_test) > 0 else 0.0

auc_test = roc_auc_score(y_test, y_test_proba)

print(f"\n📊 Test Set Performance (Threshold = {best_threshold:.3f}):")
print(f"   AUC-ROC: {auc_test:.3f}")
print(f"\n   Abnormal Class:")
print(f"   - Precision: {precision_abnormal_test:.3f}")
print(f"   - Recall:    {recall_abnormal_test:.3f}")
print(f"   - F1:        {f1_abnormal_test:.3f}")
print(f"\n   Normal Class:")
print(f"   - Precision: {precision_normal_test:.3f}")
print(f"   - Recall:    {recall_normal_test:.3f}")
print(f"   - F1:        {f1_normal_test:.3f}")

# Comparison with Step 2 (threshold=0.5)
print("\n[7] Comparison with Step 2 (threshold=0.5)...")
y_test_pred_step2 = (y_test_proba >= 0.5).astype(int)
cm_step2 = confusion_matrix(y_test, y_test_pred_step2)
tn_step2, fp_step2, fn_step2, tp_step2 = cm_step2.ravel()

precision_abnormal_step2 = tn_step2 / (tn_step2 + fn_step2) if (tn_step2 + fn_step2) > 0 else 0.0
recall_abnormal_step2 = tn_step2 / (tn_step2 + fp_step2) if (tn_step2 + fp_step2) > 0 else 0.0

comparison = pd.DataFrame([
    {
        'Method': 'Step 2 (threshold=0.5)',
        'Threshold': 0.5,
        'AUC': auc_test,
        'Precision_Abnormal': precision_abnormal_step2,
        'Recall_Abnormal': recall_abnormal_step2,
        'Precision_Normal': tp_step2 / (tp_step2 + fp_step2) if (tp_step2 + fp_step2) > 0 else 0.0,
        'Recall_Normal': tp_step2 / (tp_step2 + fn_step2) if (tp_step2 + fn_step2) > 0 else 0.0,
    },
    {
        'Method': f'Step 3-1 (threshold={best_threshold:.3f})',
        'Threshold': best_threshold,
        'AUC': auc_test,
        'Precision_Abnormal': precision_abnormal_test,
        'Recall_Abnormal': recall_abnormal_test,
        'Precision_Normal': precision_normal_test,
        'Recall_Normal': recall_normal_test,
    }
])

comparison.to_csv(OUTPUT_DIR / "comparison_step2_vs_step3_1.csv", index=False)
print(comparison.to_string(index=False))

# Save summary
summary = f"""# Step 3-1: Threshold Tuning Summary

## Goal
- Tune XGBoost classification threshold to achieve Recall ≥ 0.80 while maintaining Precision ≥ 0.85
- Tune on Validation set, apply to Test set ONCE

## Methodology
1. Load trained XGBoost model from Step 2
2. Get prediction probabilities on Validation set
3. Test thresholds from 0.05 to 0.95 (37 values)
4. Select best threshold meeting criteria (Recall ≥ 0.80 AND Precision ≥ 0.85)
5. Apply selected threshold to Test set

## Results

### Selected Threshold: {best_threshold:.3f}

### Validation Performance (Selection Criteria)
- **Precision (Abnormal)**: {best_row['precision_abnormal']:.3f}
- **Recall (Abnormal)**: {best_row['recall_abnormal']:.3f}
- **F1 (Abnormal)**: {best_row['f1_abnormal']:.3f}
- **Precision (Normal)**: {best_row['precision_normal']:.3f}
- **Recall (Normal)**: {best_row['recall_normal']:.3f}

### Test Performance
- **AUC-ROC**: {auc_test:.3f}
- **Abnormal Class**:
  - Precision: {precision_abnormal_test:.3f}
  - Recall: {recall_abnormal_test:.3f}
  - F1: {f1_abnormal_test:.3f}
- **Normal Class**:
  - Precision: {precision_normal_test:.3f}
  - Recall: {recall_normal_test:.3f}
  - F1: {f1_normal_test:.3f}

### Comparison with Step 2 (threshold=0.5)
| Metric | Step 2 | Step 3-1 | Change |
|--------|--------|----------|--------|
| Threshold | 0.500 | {best_threshold:.3f} | {best_threshold - 0.5:+.3f} |
| Precision (Abnormal) | {precision_abnormal_step2:.3f} | {precision_abnormal_test:.3f} | {precision_abnormal_test - precision_abnormal_step2:+.3f} |
| Recall (Abnormal) | {recall_abnormal_step2:.3f} | {recall_abnormal_test:.3f} | {recall_abnormal_test - recall_abnormal_step2:+.3f} |

## Key Findings
1. **Threshold Selection**: {'✅ Successfully met both criteria' if len(candidates) > 0 else '⚠️ Could not meet both criteria simultaneously'}
2. **Recall Improvement**: {f'{(recall_abnormal_test - recall_abnormal_step2) / recall_abnormal_step2 * 100:+.1f}%' if recall_abnormal_step2 > 0 else 'N/A'}
3. **Precision Trade-off**: {f'{(precision_abnormal_test - precision_abnormal_step2) / precision_abnormal_step2 * 100:+.1f}%' if precision_abnormal_step2 > 0 else 'N/A'}

## Files Generated
- `threshold_sweep_validation.csv`: Complete threshold sweep on Validation set
- `comparison_step2_vs_step3_1.csv`: Comparison with Step 2 baseline
- `step3_1_summary.md`: This summary report

## Next Steps
- Proceed to Step 3-2: Implement Hybrid Rule (XGBoost + RMS)
- Analyze which cases Hybrid catches that XGBoost misses
"""

with open(OUTPUT_DIR / "step3_1_summary.md", 'w') as f:
    f.write(summary)

print(f"\n✅ Summary saved to {OUTPUT_DIR / 'step3_1_summary.md'}")
print("\n" + "="*80)
print("STEP 3-1 COMPLETE")
print("="*80)
