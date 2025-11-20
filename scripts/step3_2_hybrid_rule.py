"""
Step 3-2: Hybrid Rule (XGBoost + RMS)
=====================================
Goal: Combine XGBoost predictions with simple RMS threshold rule
      to improve Recall while maintaining reasonable Precision

Strategy:
- (A) XGBoost alone (threshold=0.5)
- (B) RMS rule alone (acc_Y_rms > 0.15)
- (C) Hybrid: pred_hybrid = pred_xgb | pred_rms

Analysis Focus:
- Which cases does Hybrid catch that XGBoost misses?
- What is the Recall/Precision trade-off?
- Is the improvement worth the added complexity?

Key Decision:
Using threshold=0.5 (not 0.625) for research/comparison baseline
to better evaluate Hybrid's complementary effect.
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import confusion_matrix, roc_auc_score, classification_report
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODEL_DIR = PROJECT_ROOT / "docs" / "phase2_results" / "step2_xgboost_baseline"
OUTPUT_DIR = PROJECT_ROOT / "docs" / "phase2_results" / "step3_2_hybrid_rule"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Feature set (same as Step 2)
feature_set = [
    "acc_Y_rms", "acc_X_rms", "Gyro_Y_rms", "Gyro_X_rms",  # RMS core
    "acc_Y_peak", "acc_Sum_peak",  # Peak
    "acc_Y_crest",  # Crest factor
    "acc_Y_kurtosis", "acc_Sum_kurtosis",  # Kurtosis
]

# Thresholds
XGB_THRESHOLD = 0.5  # Research baseline (not 0.625)
RMS_THRESHOLD = 0.15  # From Step 1 best balanced threshold

print("="*80)
print("STEP 3-2: Hybrid Rule (XGBoost + RMS)")
print("="*80)
print(f"\nConfiguration:")
print(f"  XGBoost Threshold: {XGB_THRESHOLD}")
print(f"  RMS Threshold: {RMS_THRESHOLD} (acc_Y_rms)")
print(f"  Hybrid Strategy: pred_xgb | pred_rms")

# ============================================================================
# 1. Load Data and Model
# ============================================================================
print("\n[1] Loading data...")
features = pd.read_parquet(DATA_DIR / 'features_combined_v1.parquet')
features_clean = features.dropna()

train_df = features_clean[features_clean['split_set'] == 'train'].copy()
val_df = features_clean[features_clean['split_set'] == 'val'].copy()
test_df = features_clean[features_clean['split_set'] == 'test'].copy()

print(f"Train: {len(train_df)} samples")
print(f"Val:   {len(val_df)} samples")
print(f"Test:  {len(test_df)} samples")

# Load XGBoost model
print("\n[2] Loading XGBoost model...")
model_path = MODEL_DIR / "xgboost_baseline_v1.json"
bst = xgb.Booster()
bst.load_model(str(model_path))
print("Model loaded.")

# ============================================================================
# 2. Get Predictions for All Datasets
# ============================================================================
print("\n[3] Generating predictions for all datasets...")

def get_predictions(df, dataset_name):
    """Get XGBoost probabilities and RMS values"""
    X = df[feature_set].values
    y_true = df['label_binary'].values

    # XGBoost predictions
    dmatrix = xgb.DMatrix(X, feature_names=feature_set)
    xgb_prob = bst.predict(dmatrix)

    # RMS values
    rms_value = df['acc_Y_rms'].values

    # Create prediction dataframe
    # Note: label_binary: 1=Normal, 0=Abnormal
    # pred: 1=Normal, 0=Abnormal
    pred_df = pd.DataFrame({
        'window_id': df['window_id'].values,
        'label_binary': y_true,
        'xgb_prob': xgb_prob,
        'acc_Y_rms': rms_value,
        'pred_xgb': (xgb_prob >= XGB_THRESHOLD).astype(int),  # prob >= 0.5 → Normal (1)
        'pred_rms': (rms_value <= RMS_THRESHOLD).astype(int),  # rms <= 0.15 → Normal (1), rms > 0.15 → Abnormal (0)
    })

    # Hybrid: OR logic for Abnormal detection
    # Predict Abnormal (0) if EITHER model predicts Abnormal
    # pred_abnormal = (xgb predicts Abnormal) OR (rms predicts Abnormal)
    # pred_abnormal = (pred_xgb == 0) OR (pred_rms == 0)
    # pred_normal = NOT pred_abnormal
    pred_df['pred_hybrid'] = ((pred_df['pred_xgb'] == 1) & (pred_df['pred_rms'] == 1)).astype(int)

    return pred_df

train_pred = get_predictions(train_df, 'Train')
val_pred = get_predictions(val_df, 'Val')
test_pred = get_predictions(test_df, 'Test')

print("Predictions generated for Train, Val, Test sets.")

# ============================================================================
# 3. Evaluate All Three Strategies
# ============================================================================
print("\n[4] Evaluating strategies on Test set...")

def evaluate_strategy(pred_df, strategy_name, pred_column):
    """Evaluate a prediction strategy"""
    y_true = pred_df['label_binary'].values
    y_pred = pred_df[pred_column].values

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    # Metrics for Abnormal class (label=0, predicted=0)
    precision_abnormal = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    recall_abnormal = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    f1_abnormal = 2 * precision_abnormal * recall_abnormal / (precision_abnormal + recall_abnormal) if (precision_abnormal + recall_abnormal) > 0 else 0.0

    # Metrics for Normal class (label=1, predicted=1)
    precision_normal = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall_normal = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1_normal = 2 * precision_normal * recall_normal / (precision_normal + recall_normal) if (precision_normal + recall_normal) > 0 else 0.0

    # AUC (if probabilities available)
    if 'xgb_prob' in pred_df.columns and strategy_name == 'XGBoost':
        auc = roc_auc_score(y_true, pred_df['xgb_prob'].values)
    else:
        auc = None

    return {
        'Strategy': strategy_name,
        'TN': tn, 'FP': fp, 'FN': fn, 'TP': tp,
        'Precision_Abnormal': precision_abnormal,
        'Recall_Abnormal': recall_abnormal,
        'F1_Abnormal': f1_abnormal,
        'Precision_Normal': precision_normal,
        'Recall_Normal': recall_normal,
        'F1_Normal': f1_normal,
        'AUC': auc if auc else np.nan,
    }

# Evaluate all three strategies
results_test = []
results_test.append(evaluate_strategy(test_pred, 'XGBoost (τ=0.5)', 'pred_xgb'))
results_test.append(evaluate_strategy(test_pred, 'RMS (T=0.15)', 'pred_rms'))
results_test.append(evaluate_strategy(test_pred, 'Hybrid (XGB | RMS)', 'pred_hybrid'))

results_df = pd.DataFrame(results_test)

# Save results
results_df.to_csv(OUTPUT_DIR / "strategy_comparison_test.csv", index=False)

print("\n" + "="*80)
print("STRATEGY COMPARISON - TEST SET")
print("="*80)
print(results_df.to_string(index=False))

# ============================================================================
# 4. Analyze Hybrid Improvement Cases
# ============================================================================
print("\n[5] Analyzing cases where Hybrid improves over XGBoost...")

# Find cases where Hybrid catches but XGBoost misses
# True Abnormal (label=0) that XGBoost predicts as Normal (pred_xgb=1) but Hybrid catches (pred_hybrid=0)
improved_cases = test_pred[
    (test_pred['label_binary'] == 0) &  # True Abnormal
    (test_pred['pred_xgb'] == 1) &       # XGBoost missed (predicted Normal)
    (test_pred['pred_hybrid'] == 0)      # Hybrid caught (predicted Abnormal)
].copy()

print(f"\n✅ Hybrid caught {len(improved_cases)} additional True Abnormal cases that XGBoost missed")

if len(improved_cases) > 0:
    print("\nImproved Case Statistics:")
    print(f"  acc_Y_rms: mean={improved_cases['acc_Y_rms'].mean():.4f}, "
          f"median={improved_cases['acc_Y_rms'].median():.4f}, "
          f"min={improved_cases['acc_Y_rms'].min():.4f}, "
          f"max={improved_cases['acc_Y_rms'].max():.4f}")
    print(f"  xgb_prob:  mean={improved_cases['xgb_prob'].mean():.4f}, "
          f"median={improved_cases['xgb_prob'].median():.4f}, "
          f"min={improved_cases['xgb_prob'].min():.4f}, "
          f"max={improved_cases['xgb_prob'].max():.4f}")

    # Save improved cases
    improved_cases.to_csv(OUTPUT_DIR / "hybrid_improved_cases.csv", index=False)
    print(f"\n  Saved to: hybrid_improved_cases.csv")

# Find cases where Hybrid adds False Positives
# True Normal (label=1) that XGBoost correctly predicts as Normal (pred_xgb=1) but Hybrid wrongly flags (pred_hybrid=0)
added_fp_cases = test_pred[
    (test_pred['label_binary'] == 1) &  # True Normal
    (test_pred['pred_xgb'] == 1) &       # XGBoost correct (predicted Normal)
    (test_pred['pred_hybrid'] == 0)      # Hybrid wrong (predicted Abnormal)
].copy()

print(f"\n⚠️  Hybrid added {len(added_fp_cases)} False Positives (Normal → flagged as Abnormal)")

if len(added_fp_cases) > 0:
    print("\nAdded FP Case Statistics:")
    print(f"  acc_Y_rms: mean={added_fp_cases['acc_Y_rms'].mean():.4f}, "
          f"median={added_fp_cases['acc_Y_rms'].median():.4f}, "
          f"min={added_fp_cases['acc_Y_rms'].min():.4f}, "
          f"max={added_fp_cases['acc_Y_rms'].max():.4f}")
    print(f"  xgb_prob:  mean={added_fp_cases['xgb_prob'].mean():.4f}, "
          f"median={added_fp_cases['xgb_prob'].median():.4f}, "
          f"min={added_fp_cases['xgb_prob'].min():.4f}, "
          f"max={added_fp_cases['xgb_prob'].max():.4f}")

    # Save added FP cases
    added_fp_cases.to_csv(OUTPUT_DIR / "hybrid_added_fp_cases.csv", index=False)
    print(f"  Saved to: hybrid_added_fp_cases.csv")

# ============================================================================
# 5. Visualization: Scatter Plot (xgb_prob vs acc_Y_rms)
# ============================================================================
print("\n[6] Generating visualization...")

fig, ax = plt.subplots(figsize=(10, 8))

# Separate by true label
abnormal_mask = test_pred['label_binary'] == 0
normal_mask = test_pred['label_binary'] == 1

# Plot Abnormal samples
ax.scatter(
    test_pred[abnormal_mask]['xgb_prob'],
    test_pred[abnormal_mask]['acc_Y_rms'],
    c='red', marker='o', s=50, alpha=0.6, label='Abnormal (True)'
)

# Plot Normal samples
ax.scatter(
    test_pred[normal_mask]['xgb_prob'],
    test_pred[normal_mask]['acc_Y_rms'],
    c='blue', marker='s', s=50, alpha=0.6, label='Normal (True)'
)

# Highlight Hybrid improvement cases
if len(improved_cases) > 0:
    ax.scatter(
        improved_cases['xgb_prob'],
        improved_cases['acc_Y_rms'],
        c='green', marker='^', s=150, alpha=0.9,
        edgecolors='black', linewidths=2,
        label=f'Hybrid Caught ({len(improved_cases)} cases)'
    )

# Decision boundaries
ax.axvline(x=XGB_THRESHOLD, color='purple', linestyle='--', linewidth=2, label=f'XGB Threshold ({XGB_THRESHOLD})')
ax.axhline(y=RMS_THRESHOLD, color='orange', linestyle='--', linewidth=2, label=f'RMS Threshold ({RMS_THRESHOLD})')

# Quadrant labels
ax.text(0.25, 0.22, 'Both\nAbnormal', ha='center', va='center', fontsize=10, color='gray', alpha=0.7)
ax.text(0.75, 0.22, 'XGB Normal\nRMS Abnormal', ha='center', va='center', fontsize=10, color='gray', alpha=0.7)
ax.text(0.25, 0.08, 'XGB Abnormal\nRMS Normal', ha='center', va='center', fontsize=10, color='gray', alpha=0.7)
ax.text(0.75, 0.08, 'Both\nNormal', ha='center', va='center', fontsize=10, color='gray', alpha=0.7)

ax.set_xlabel('XGBoost Probability (Normal)', fontsize=12)
ax.set_ylabel('acc_Y_rms', fontsize=12)
ax.set_title('Hybrid Rule Decision Space\n(Test Set)', fontsize=14, fontweight='bold')
ax.legend(loc='upper left', fontsize=10)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "hybrid_decision_space.png", dpi=150, bbox_inches='tight')
plt.close()

print(f"Visualization saved to: hybrid_decision_space.png")

# Export scatter data
scatter_data = test_pred[['window_id', 'label_binary', 'xgb_prob', 'acc_Y_rms', 'pred_xgb', 'pred_rms', 'pred_hybrid']].copy()
scatter_data.to_csv(OUTPUT_DIR / "hybrid_scatter_data.csv", index=False)
print(f"Scatter data saved to: hybrid_scatter_data.csv")

# ============================================================================
# 6. Summary Report
# ============================================================================
print("\n[7] Generating summary report...")

# Calculate improvements
xgb_recall = results_df[results_df['Strategy'] == 'XGBoost (τ=0.5)']['Recall_Abnormal'].values[0]
hybrid_recall = results_df[results_df['Strategy'] == 'Hybrid (XGB | RMS)']['Recall_Abnormal'].values[0]
recall_improvement = hybrid_recall - xgb_recall

xgb_precision = results_df[results_df['Strategy'] == 'XGBoost (τ=0.5)']['Precision_Abnormal'].values[0]
hybrid_precision = results_df[results_df['Strategy'] == 'Hybrid (XGB | RMS)']['Precision_Abnormal'].values[0]
precision_change = hybrid_precision - xgb_precision

summary = f"""# Step 3-2: Hybrid Rule Summary

## Goal
Combine XGBoost predictions with RMS threshold rule to improve Recall
while evaluating the Precision trade-off.

## Configuration
- **XGBoost Threshold**: {XGB_THRESHOLD} (research baseline, not 0.625)
- **RMS Threshold**: {RMS_THRESHOLD} (acc_Y_rms, from Step 1)
- **Hybrid Strategy**: `pred_hybrid = pred_xgb | pred_rms`
  - Predict Abnormal if EITHER model predicts Abnormal

## Rationale for threshold=0.5
Using threshold=0.5 (not 0.625) allows us to:
1. Evaluate Hybrid's complementary effect more clearly
2. Maintain balanced Normal/Abnormal detection
3. Use this as a fair research baseline for future comparisons

Step 3-1's threshold=0.625 is reserved as an "ultra-conservative mode" option.

## Results - Strategy Comparison (Test Set)

| Strategy | Precision (Abn) | Recall (Abn) | F1 (Abn) | Precision (Norm) | Recall (Norm) | F1 (Norm) |
|----------|-----------------|--------------|----------|------------------|---------------|-----------|
| XGBoost (τ=0.5) | {results_df.iloc[0]['Precision_Abnormal']:.3f} | {results_df.iloc[0]['Recall_Abnormal']:.3f} | {results_df.iloc[0]['F1_Abnormal']:.3f} | {results_df.iloc[0]['Precision_Normal']:.3f} | {results_df.iloc[0]['Recall_Normal']:.3f} | {results_df.iloc[0]['F1_Normal']:.3f} |
| RMS (T=0.15) | {results_df.iloc[1]['Precision_Abnormal']:.3f} | {results_df.iloc[1]['Recall_Abnormal']:.3f} | {results_df.iloc[1]['F1_Abnormal']:.3f} | {results_df.iloc[1]['Precision_Normal']:.3f} | {results_df.iloc[1]['Recall_Normal']:.3f} | {results_df.iloc[1]['F1_Normal']:.3f} |
| Hybrid (XGB \\| RMS) | {results_df.iloc[2]['Precision_Abnormal']:.3f} | {results_df.iloc[2]['Recall_Abnormal']:.3f} | {results_df.iloc[2]['F1_Abnormal']:.3f} | {results_df.iloc[2]['Precision_Normal']:.3f} | {results_df.iloc[2]['Recall_Normal']:.3f} | {results_df.iloc[2]['F1_Normal']:.3f} |

### Key Metrics Change (XGBoost → Hybrid)
- **Recall (Abnormal)**: {xgb_recall:.3f} → {hybrid_recall:.3f} ({recall_improvement:+.3f}, {recall_improvement/xgb_recall*100:+.1f}%)
- **Precision (Abnormal)**: {xgb_precision:.3f} → {hybrid_precision:.3f} ({precision_change:+.3f}, {precision_change/xgb_precision*100:+.1f}%)

## Hybrid Improvement Analysis

### ✅ True Positives Added
Hybrid caught **{len(improved_cases)} additional True Abnormal cases** that XGBoost missed.

{'**Characteristics of Improved Cases:**' if len(improved_cases) > 0 else ''}
{f"- acc_Y_rms: mean={improved_cases['acc_Y_rms'].mean():.4f}, median={improved_cases['acc_Y_rms'].median():.4f}" if len(improved_cases) > 0 else ''}
{f"- xgb_prob: mean={improved_cases['xgb_prob'].mean():.4f}, median={improved_cases['xgb_prob'].median():.4f}" if len(improved_cases) > 0 else ''}
{f"- Interpretation: XGBoost gave these cases Normal probabilities (prob >= 0.5), but RMS rule caught them (acc_Y_rms > 0.15)" if len(improved_cases) > 0 else ''}

### ⚠️ False Positives Added
Hybrid added **{len(added_fp_cases)} False Positives** (Normal samples flagged as Abnormal).

{'**Characteristics of Added FP Cases:**' if len(added_fp_cases) > 0 else ''}
{f"- acc_Y_rms: mean={added_fp_cases['acc_Y_rms'].mean():.4f}, median={added_fp_cases['acc_Y_rms'].median():.4f}" if len(added_fp_cases) > 0 else ''}
{f"- xgb_prob: mean={added_fp_cases['xgb_prob'].mean():.4f}, median={added_fp_cases['xgb_prob'].median():.4f}" if len(added_fp_cases) > 0 else ''}
{f"- Interpretation: XGBoost correctly classified these as Normal, but RMS rule flagged them (acc_Y_rms > 0.15)" if len(added_fp_cases) > 0 else ''}

## Key Findings

1. **Hybrid Effectiveness**: {'Hybrid successfully improves Recall' if recall_improvement > 0 else 'Hybrid does not improve Recall significantly'}
2. **Precision Trade-off**: {'Acceptable precision loss' if abs(precision_change) < 0.1 else 'Significant precision impact'}
3. **Complementary Value**: {'RMS rule complements XGBoost by catching cases with high vibration that XGBoost misses' if len(improved_cases) > 0 else 'Limited complementary value observed'}

## Files Generated
- `strategy_comparison_test.csv`: Performance metrics for all three strategies
- `hybrid_improved_cases.csv`: Cases where Hybrid caught XGBoost misses
- `hybrid_added_fp_cases.csv`: False positives added by Hybrid
- `hybrid_decision_space.png`: Visualization of decision boundaries
- `hybrid_scatter_data.csv`: Scatter plot data (xgb_prob vs acc_Y_rms)
- `step3_2_summary.md`: This summary report

## Next Steps
- Proceed to Step 3-3: Add Band RMS features (1-10Hz, 10-50Hz, 50-150Hz)
- Re-train XGBoost with expanded feature set
- Evaluate if better features reduce need for extreme thresholds or hybrid rules
"""

with open(OUTPUT_DIR / "step3_2_summary.md", 'w') as f:
    f.write(summary)

print(f"Summary saved to: step3_2_summary.md")

print("\n" + "="*80)
print("STEP 3-2 COMPLETE")
print("="*80)
print(f"\nKey Result:")
print(f"  Hybrid Recall improvement: {recall_improvement:+.3f} ({recall_improvement/xgb_recall*100:+.1f}%)")
print(f"  Hybrid Precision change: {precision_change:+.3f} ({precision_change/xgb_precision*100:+.1f}%)")
print(f"  Additional TP caught: {len(improved_cases)}")
print(f"  Additional FP added: {len(added_fp_cases)}")
