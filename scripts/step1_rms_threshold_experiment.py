"""
Step 1: RMS Threshold Rule Experiment
간단한 RMS threshold만으로 어느 정도 분리되는지 확인

목적:
1. acc_Y_rms 하나만 사용하여 threshold 탐색
2. 각 threshold에서 성능 지표 계산
3. 하이브리드 룰 설계 기준선 확보
4. XGBoost/AE와 비교할 baseline 성능 확보
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import (
    confusion_matrix, precision_recall_fscore_support,
    roc_curve, auc, precision_recall_curve
)

# Paths
BASE_PATH = Path(__file__).parent.parent
DATA_PATH = BASE_PATH / "data" / "processed"
OUTPUT_PATH = BASE_PATH / "docs" / "phase2_results" / "step1_rms_rule"
OUTPUT_PATH.mkdir(exist_ok=True, parents=True)

print("=" * 80)
print("Step 1: RMS Threshold Rule Experiment")
print("=" * 80)
print()

# ============================================================================
# 1. Load Data
# ============================================================================
print("[1/5] Loading data...")
features = pd.read_parquet(DATA_PATH / 'features_combined_v1.parquet')
features_clean = features.dropna()

print(f"  Total windows: {len(features)}")
print(f"  After removing NaN: {len(features_clean)}")
print()

# ============================================================================
# 2. Split by dataset
# ============================================================================
print("[2/5] Splitting by dataset...")
train_df = features_clean[features_clean['split_set'] == 'train'].copy()
val_df = features_clean[features_clean['split_set'] == 'val'].copy()
test_df = features_clean[features_clean['split_set'] == 'test'].copy()

print(f"  Train: {len(train_df)} (Normal: {(train_df['label_binary']==1).sum()}, Abnormal: {(train_df['label_binary']==0).sum()})")
print(f"  Val:   {len(val_df)} (Normal: {(val_df['label_binary']==1).sum()}, Abnormal: {(val_df['label_binary']==0).sum()})")
print(f"  Test:  {len(test_df)} (Normal: {(test_df['label_binary']==1).sum()}, Abnormal: {(test_df['label_binary']==0).sum()})")
print()

# ============================================================================
# 3. RMS Threshold Experiment
# ============================================================================
print("[3/5] RMS Threshold experiment on acc_Y_rms...")

# Threshold candidates
thresholds = [0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.60, 0.70, 0.80]

# Function to evaluate threshold
def evaluate_threshold(df, feature, threshold, dataset_name):
    """Evaluate a single threshold on a dataset"""
    y_true = df['label_binary'].values
    y_pred = (df[feature] > threshold).astype(int)

    # Note: label_binary=1 is Normal, 0 is Abnormal
    # Rule: feature > threshold → Abnormal (predict 0)
    # So we need to invert for sklearn which expects 1 as positive
    y_pred_inv = 1 - y_pred  # Now 1=Normal, 0=Abnormal

    cm = confusion_matrix(y_true, y_pred_inv)
    tn, fp, fn, tp = cm.ravel()

    # Calculate metrics (treating Normal=1 as positive)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    # For Abnormal detection
    abnormal_precision = tn / (tn + fn) if (tn + fn) > 0 else 0
    abnormal_recall = tn / (tn + fp) if (tn + fp) > 0 else 0
    abnormal_f1 = 2 * abnormal_precision * abnormal_recall / (abnormal_precision + abnormal_recall) if (abnormal_precision + abnormal_recall) > 0 else 0

    return {
        'dataset': dataset_name,
        'threshold': threshold,
        'tn': tn,  # True Abnormal
        'fp': fp,  # False Abnormal (Normal predicted as Abnormal)
        'fn': fn,  # False Normal (Abnormal predicted as Normal)
        'tp': tp,  # True Normal
        'accuracy': (tp + tn) / (tp + tn + fp + fn),
        'normal_precision': precision,
        'normal_recall': recall,
        'normal_f1': f1,
        'abnormal_precision': abnormal_precision,
        'abnormal_recall': abnormal_recall,
        'abnormal_f1': abnormal_f1,
    }

# Evaluate on all datasets
results = []
for threshold in thresholds:
    results.append(evaluate_threshold(train_df, 'acc_Y_rms', threshold, 'train'))
    results.append(evaluate_threshold(val_df, 'acc_Y_rms', threshold, 'val'))
    results.append(evaluate_threshold(test_df, 'acc_Y_rms', threshold, 'test'))

results_df = pd.DataFrame(results)
results_df.to_csv(OUTPUT_PATH / 'rms_threshold_analysis.csv', index=False)
print(f"  ✓ Saved: rms_threshold_analysis.csv")
print()

# ============================================================================
# 4. Find best thresholds
# ============================================================================
print("[4/5] Finding best thresholds...")

# Best threshold for each metric (on validation set)
val_results = results_df[results_df['dataset'] == 'val'].copy()

best_f1_idx = val_results['abnormal_f1'].idxmax()
best_recall_idx = val_results['abnormal_recall'].idxmax()
best_precision_idx = val_results['abnormal_precision'].idxmax()

print("\n  Best thresholds on Validation set:")
print(f"  Best F1 (Abnormal):        {val_results.loc[best_f1_idx, 'threshold']:.2f} (F1={val_results.loc[best_f1_idx, 'abnormal_f1']:.3f})")
print(f"  Best Recall (Abnormal):    {val_results.loc[best_recall_idx, 'threshold']:.2f} (Recall={val_results.loc[best_recall_idx, 'abnormal_recall']:.3f})")
print(f"  Best Precision (Abnormal): {val_results.loc[best_precision_idx, 'threshold']:.2f} (Precision={val_results.loc[best_precision_idx, 'abnormal_precision']:.3f})")

# Best balanced threshold (Precision >= 0.7, maximize Recall)
balanced_candidates = val_results[val_results['abnormal_precision'] >= 0.7]
if len(balanced_candidates) > 0:
    best_balanced_idx = balanced_candidates['abnormal_recall'].idxmax()
    print(f"  Best Balanced (P≥0.7):     {balanced_candidates.loc[best_balanced_idx, 'threshold']:.2f} (P={balanced_candidates.loc[best_balanced_idx, 'abnormal_precision']:.3f}, R={balanced_candidates.loc[best_balanced_idx, 'abnormal_recall']:.3f})")
else:
    print("  No threshold meets Precision >= 0.7 criterion")

print()

# ============================================================================
# 5. ROC Curve Analysis
# ============================================================================
print("[5/5] ROC Curve analysis...")

# Calculate ROC for each dataset
roc_data = []
for dataset_name, df in [('train', train_df), ('val', val_df), ('test', test_df)]:
    y_true = df['label_binary'].values
    y_score = df['acc_Y_rms'].values

    # Since higher RMS = Abnormal (0), we need to negate for ROC
    # Or we can use 1 - label_binary to treat Abnormal as positive
    y_true_abnormal = 1 - y_true  # Now 1=Abnormal, 0=Normal

    fpr, tpr, thresholds_roc = roc_curve(y_true_abnormal, y_score)
    roc_auc = auc(fpr, tpr)

    for i, (f, t, th) in enumerate(zip(fpr, tpr, thresholds_roc)):
        roc_data.append({
            'dataset': dataset_name,
            'fpr': f,
            'tpr': t,
            'threshold': th,
            'auc': roc_auc
        })

roc_df = pd.DataFrame(roc_data)
roc_df.to_csv(OUTPUT_PATH / 'rms_roc_curve_data.csv', index=False)
print(f"  ✓ Saved: rms_roc_curve_data.csv")

# Print AUC
for dataset_name in ['train', 'val', 'test']:
    dataset_auc = roc_df[roc_df['dataset'] == dataset_name]['auc'].iloc[0]
    print(f"  {dataset_name.capitalize()} AUC: {dataset_auc:.4f}")

print()

# ============================================================================
# 6. Summary Report
# ============================================================================
print("Generating summary report...")

summary_lines = []
summary_lines.append("# Step 1: RMS Threshold Rule Experiment - Summary Report")
summary_lines.append("")
summary_lines.append("**Date**: 2025-11-17")
summary_lines.append("**Feature**: acc_Y_rms")
summary_lines.append("**Thresholds tested**: " + str(thresholds))
summary_lines.append("")
summary_lines.append("---")
summary_lines.append("")
summary_lines.append("## 1. Dataset Statistics")
summary_lines.append("")
summary_lines.append(f"| Dataset | Total | Normal | Abnormal | Ratio (N:A) |")
summary_lines.append(f"|---------|-------|--------|----------|-------------|")
summary_lines.append(f"| Train | {len(train_df)} | {(train_df['label_binary']==1).sum()} | {(train_df['label_binary']==0).sum()} | 1:{(train_df['label_binary']==0).sum()/(train_df['label_binary']==1).sum():.1f} |")
summary_lines.append(f"| Val | {len(val_df)} | {(val_df['label_binary']==1).sum()} | {(val_df['label_binary']==0).sum()} | 1:{(val_df['label_binary']==0).sum()/(val_df['label_binary']==1).sum():.1f} |")
summary_lines.append(f"| Test | {len(test_df)} | {(test_df['label_binary']==1).sum()} | {(test_df['label_binary']==0).sum()} | 1:{(test_df['label_binary']==0).sum()/(test_df['label_binary']==1).sum():.1f} |")
summary_lines.append("")
summary_lines.append("---")
summary_lines.append("")
summary_lines.append("## 2. Best Thresholds (Validation Set)")
summary_lines.append("")
summary_lines.append("| Metric | Threshold | Value |")
summary_lines.append("|--------|-----------|-------|")
summary_lines.append(f"| Best F1 (Abnormal) | {val_results.loc[best_f1_idx, 'threshold']:.2f} | {val_results.loc[best_f1_idx, 'abnormal_f1']:.3f} |")
summary_lines.append(f"| Best Recall (Abnormal) | {val_results.loc[best_recall_idx, 'threshold']:.2f} | {val_results.loc[best_recall_idx, 'abnormal_recall']:.3f} |")
summary_lines.append(f"| Best Precision (Abnormal) | {val_results.loc[best_precision_idx, 'threshold']:.2f} | {val_results.loc[best_precision_idx, 'abnormal_precision']:.3f} |")

if len(balanced_candidates) > 0:
    summary_lines.append(f"| **Best Balanced (P≥0.7)** | **{balanced_candidates.loc[best_balanced_idx, 'threshold']:.2f}** | **P={balanced_candidates.loc[best_balanced_idx, 'abnormal_precision']:.3f}, R={balanced_candidates.loc[best_balanced_idx, 'abnormal_recall']:.3f}** |")

summary_lines.append("")
summary_lines.append("---")
summary_lines.append("")
summary_lines.append("## 3. ROC AUC Scores")
summary_lines.append("")
summary_lines.append("| Dataset | AUC |")
summary_lines.append("|---------|-----|")
for dataset_name in ['train', 'val', 'test']:
    dataset_auc = roc_df[roc_df['dataset'] == dataset_name]['auc'].iloc[0]
    summary_lines.append(f"| {dataset_name.capitalize()} | {dataset_auc:.4f} |")

summary_lines.append("")
summary_lines.append("---")
summary_lines.append("")
summary_lines.append("## 4. Key Findings")
summary_lines.append("")
summary_lines.append("### Strengths")
summary_lines.append("- Simple threshold rule provides baseline performance")
summary_lines.append("- acc_Y_rms alone achieves meaningful separation")
summary_lines.append("- Interpretable decision boundary")
summary_lines.append("")
summary_lines.append("### Limitations")
summary_lines.append("- Single threshold cannot capture complex patterns")
summary_lines.append("- Trade-off between Precision and Recall")
summary_lines.append("- Cannot leverage multiple features")
summary_lines.append("")
summary_lines.append("---")
summary_lines.append("")
summary_lines.append("## 5. Next Steps")
summary_lines.append("")
summary_lines.append("**Step 2**: XGBoost Baseline")
summary_lines.append("- Use 9 features (RMS core + Peak + Crest + Kurtosis)")
summary_lines.append("- Target: CV AUC > 0.75, Test AUC > 0.70")
summary_lines.append("- Compare with Step 1 baseline")
summary_lines.append("")
summary_lines.append("**Files Generated**:")
summary_lines.append("- `rms_threshold_analysis.csv`: Detailed threshold analysis")
summary_lines.append("- `rms_roc_curve_data.csv`: ROC curve data for plotting")
summary_lines.append("- `rms_rule_summary.md`: This summary report")

with open(OUTPUT_PATH / 'rms_rule_summary.md', 'w', encoding='utf-8') as f:
    f.write('\n'.join(summary_lines))

print(f"  ✓ Saved: rms_rule_summary.md")
print()

# ============================================================================
# Final Report
# ============================================================================
print("=" * 80)
print("Step 1 Complete!")
print("=" * 80)
print()
print(f"Output directory: {OUTPUT_PATH}")
print()
print("Files generated:")
print("  1. rms_threshold_analysis.csv - Threshold performance metrics")
print("  2. rms_roc_curve_data.csv - ROC curve data")
print("  3. rms_rule_summary.md - Summary report")
print()
print("=" * 80)
print("Key Metrics (Best Balanced Threshold on Val):")
print("=" * 80)

if len(balanced_candidates) > 0:
    best_threshold = balanced_candidates.loc[best_balanced_idx, 'threshold']
    best_p = balanced_candidates.loc[best_balanced_idx, 'abnormal_precision']
    best_r = balanced_candidates.loc[best_balanced_idx, 'abnormal_recall']
    best_f1 = balanced_candidates.loc[best_balanced_idx, 'abnormal_f1']

    print(f"Threshold: {best_threshold:.2f}")
    print(f"Precision (Abnormal): {best_p:.3f}")
    print(f"Recall (Abnormal): {best_r:.3f}")
    print(f"F1 (Abnormal): {best_f1:.3f}")

    # Evaluate on test set with best threshold
    test_eval = evaluate_threshold(test_df, 'acc_Y_rms', best_threshold, 'test')
    print()
    print("Test Set Performance (with best threshold):")
    print(f"  Precision (Abnormal): {test_eval['abnormal_precision']:.3f}")
    print(f"  Recall (Abnormal): {test_eval['abnormal_recall']:.3f}")
    print(f"  F1 (Abnormal): {test_eval['abnormal_f1']:.3f}")
else:
    print("No threshold meets Precision >= 0.7 criterion")
    print("Consider lowering precision requirement or using more features")

print()
print("✅ Step 1 완료 - 다음 단계(Step 2: XGBoost Baseline)로 진행 가능합니다.")
