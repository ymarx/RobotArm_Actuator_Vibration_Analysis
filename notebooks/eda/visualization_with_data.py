"""
Visualization with Data Export
시각화 + 수치 데이터 저장

목적:
1. 2D scatter plots (acc_Sum_rms vs kurtosis, peak vs crest 등)
2. 클래스별 boxplot (RMS features)
3. 시각화에 사용된 데이터를 CSV로 저장 (AI가 분석 가능하도록)
4. RMS features 심층 분석
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt

# Paths
BASE_PATH = Path(__file__).parent.parent.parent
DATA_PATH = BASE_PATH / "data" / "processed"
OUTPUT_PATH = BASE_PATH / "claudedocs" / "eda_results"
VIZ_PATH = OUTPUT_PATH / "visualizations"
VIZ_PATH.mkdir(exist_ok=True, parents=True)

print("=" * 80)
print("시각화 + 데이터 분석")
print("=" * 80)
print()

# Load data
print("[1/6] 데이터 로딩...")
features = pd.read_parquet(DATA_PATH / 'features_combined_v1.parquet')
features_clean = features.dropna()

print(f"  Total windows: {len(features)}")
print(f"  After removing NaN: {len(features_clean)}")
print()

# Identify RMS features
rms_features = [col for col in features_clean.columns if 'rms' in col.lower()]
peak_features = [col for col in features_clean.columns if 'peak' in col.lower()]
kurtosis_features = [col for col in features_clean.columns if 'kurtosis' in col.lower()]
crest_features = [col for col in features_clean.columns if 'crest' in col.lower()]

print(f"RMS features: {len(rms_features)}")
print(f"Peak features: {len(peak_features)}")
print(f"Kurtosis features: {len(kurtosis_features)}")
print(f"Crest features: {len(crest_features)}")
print()

# ============================================================================
# 2. RMS Features Analysis
# ============================================================================
print("[2/6] RMS Features 심층 분석...")

# Train set only for fair comparison
train_df = features_clean[features_clean['split_set'] == 'train'].copy()

# Class labels
train_df['class_label'] = train_df['label_binary'].map({0: 'Abnormal', 1: 'Normal'})

# RMS features statistics by class
rms_stats_list = []
for col in rms_features:
    for label in [0, 1]:
        subset = train_df[train_df['label_binary'] == label][col]
        rms_stats_list.append({
            'feature': col,
            'class': 'Normal' if label == 1 else 'Abnormal',
            'mean': subset.mean(),
            'median': subset.median(),
            'std': subset.std(),
            'min': subset.min(),
            'q25': subset.quantile(0.25),
            'q75': subset.quantile(0.75),
            'max': subset.max(),
            'count': len(subset)
        })

rms_stats_df = pd.DataFrame(rms_stats_list)
rms_stats_df.to_csv(OUTPUT_PATH / 'rms_statistics_by_class.csv', index=False)
print(f"  ✓ Saved: rms_statistics_by_class.csv")

# RMS ratios (Abnormal / Normal)
rms_ratios = []
for col in rms_features:
    normal_mean = train_df[train_df['label_binary'] == 1][col].mean()
    abnormal_mean = train_df[train_df['label_binary'] == 0][col].mean()
    ratio = abnormal_mean / normal_mean if normal_mean > 0 else np.nan

    rms_ratios.append({
        'feature': col,
        'normal_mean': normal_mean,
        'abnormal_mean': abnormal_mean,
        'abnormal_to_normal_ratio': ratio,
        'difference': abnormal_mean - normal_mean,
        'pct_increase': (ratio - 1) * 100 if not np.isnan(ratio) else np.nan
    })

rms_ratios_df = pd.DataFrame(rms_ratios).sort_values('abnormal_to_normal_ratio', ascending=False)
rms_ratios_df.to_csv(OUTPUT_PATH / 'rms_abnormal_to_normal_ratios.csv', index=False)
print(f"  ✓ Saved: rms_abnormal_to_normal_ratios.csv")
print()

# ============================================================================
# 3. 2D Scatter Plots with Data
# ============================================================================
print("[3/6] 2D Scatter Plots 생성...")

scatter_pairs = [
    ('acc_Sum_rms', 'acc_Sum_kurtosis'),
    ('acc_Sum_rms', 'acc_Sum_peak'),
    ('acc_Y_rms', 'acc_Y_kurtosis'),
    ('Gyro_Y_rms', 'Gyro_Y_kurtosis'),
    ('acc_Sum_peak', 'acc_Sum_crest'),
    ('acc_Y_std', 'acc_Y_rms'),
]

for idx, (x_feat, y_feat) in enumerate(scatter_pairs, 1):
    if x_feat not in train_df.columns or y_feat not in train_df.columns:
        continue

    # Extract data for plotting
    plot_data = train_df[[x_feat, y_feat, 'label_binary', 'class_label', 'window_id']].copy()

    # Save plot data to CSV
    csv_name = f'scatter_data_{idx}_{x_feat}_vs_{y_feat}.csv'
    plot_data.to_csv(VIZ_PATH / csv_name, index=False)

    # Create plot
    plt.figure(figsize=(10, 8))

    for label, color, marker in [(0, 'red', 'o'), (1, 'blue', 's')]:
        subset = plot_data[plot_data['label_binary'] == label]
        plt.scatter(subset[x_feat], subset[y_feat],
                   c=color, marker=marker, alpha=0.6, s=50,
                   label='Normal' if label == 1 else 'Abnormal')

    plt.xlabel(x_feat, fontsize=12)
    plt.ylabel(y_feat, fontsize=12)
    plt.title(f'{x_feat} vs {y_feat} (Train Set)', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plot_name = f'scatter_{idx}_{x_feat}_vs_{y_feat}.png'
    plt.savefig(VIZ_PATH / plot_name, dpi=150)
    plt.close()

    print(f"  ✓ Saved: {plot_name} + {csv_name}")

print()

# ============================================================================
# 4. Boxplots for RMS Features
# ============================================================================
print("[4/6] RMS Boxplots 생성...")

# Top RMS features by class separation
top_rms = rms_ratios_df.head(7)['feature'].tolist()

for idx, feat in enumerate(top_rms, 1):
    if feat not in train_df.columns:
        continue

    # Extract boxplot data
    boxplot_data = train_df[[feat, 'class_label', 'window_id']].copy()
    csv_name = f'boxplot_data_{idx}_{feat}.csv'
    boxplot_data.to_csv(VIZ_PATH / csv_name, index=False)

    # Calculate summary statistics for annotation
    normal_vals = train_df[train_df['label_binary'] == 1][feat]
    abnormal_vals = train_df[train_df['label_binary'] == 0][feat]

    summary_stats = pd.DataFrame({
        'class': ['Normal', 'Abnormal'],
        'feature': [feat, feat],
        'count': [len(normal_vals), len(abnormal_vals)],
        'mean': [normal_vals.mean(), abnormal_vals.mean()],
        'median': [normal_vals.median(), abnormal_vals.median()],
        'q25': [normal_vals.quantile(0.25), abnormal_vals.quantile(0.25)],
        'q75': [normal_vals.quantile(0.75), abnormal_vals.quantile(0.75)],
        'min': [normal_vals.min(), abnormal_vals.min()],
        'max': [normal_vals.max(), abnormal_vals.max()],
    })
    summary_csv_name = f'boxplot_summary_{idx}_{feat}.csv'
    summary_stats.to_csv(VIZ_PATH / summary_csv_name, index=False)

    # Create boxplot
    plt.figure(figsize=(8, 6))
    boxplot_data.boxplot(column=feat, by='class_label', ax=plt.gca())
    plt.suptitle('')  # Remove default title
    plt.title(f'{feat} by Class', fontsize=14)
    plt.xlabel('Class', fontsize=12)
    plt.ylabel(feat, fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plot_name = f'boxplot_{idx}_{feat}.png'
    plt.savefig(VIZ_PATH / plot_name, dpi=150)
    plt.close()

    print(f"  ✓ Saved: {plot_name} + {csv_name} + {summary_csv_name}")

print()

# ============================================================================
# 5. Correlation Matrix for RMS Features
# ============================================================================
print("[5/6] RMS Correlation Matrix...")

rms_corr = train_df[rms_features].corr()
rms_corr_flat = []
for i, feat1 in enumerate(rms_features):
    for j, feat2 in enumerate(rms_features):
        if i < j:  # Upper triangle only
            rms_corr_flat.append({
                'feature1': feat1,
                'feature2': feat2,
                'correlation': rms_corr.loc[feat1, feat2]
            })

rms_corr_df = pd.DataFrame(rms_corr_flat).sort_values('correlation', key=abs, ascending=False)
rms_corr_df.to_csv(OUTPUT_PATH / 'rms_correlation_matrix.csv', index=False)
print(f"  ✓ Saved: rms_correlation_matrix.csv")

# High correlation pairs (|r| > 0.95)
high_corr_rms = rms_corr_df[rms_corr_df['correlation'].abs() > 0.95]
if len(high_corr_rms) > 0:
    high_corr_rms.to_csv(OUTPUT_PATH / 'rms_high_correlation_pairs.csv', index=False)
    print(f"  ✓ High correlation RMS pairs (|r|>0.95): {len(high_corr_rms)}")
print()

# ============================================================================
# 6. Summary Statistics Table
# ============================================================================
print("[6/6] 종합 통계 테이블 생성...")

summary_table = []

# Overall statistics
summary_table.append({
    'category': 'Data',
    'metric': 'Total Windows (Train)',
    'value': len(train_df)
})
summary_table.append({
    'category': 'Data',
    'metric': 'Normal Windows',
    'value': len(train_df[train_df['label_binary'] == 1])
})
summary_table.append({
    'category': 'Data',
    'metric': 'Abnormal Windows',
    'value': len(train_df[train_df['label_binary'] == 0])
})

# RMS statistics
summary_table.append({
    'category': 'RMS Features',
    'metric': 'Total RMS Features',
    'value': len(rms_features)
})
summary_table.append({
    'category': 'RMS Features',
    'metric': 'Mean Abnormal/Normal Ratio',
    'value': f"{rms_ratios_df['abnormal_to_normal_ratio'].mean():.3f}"
})
summary_table.append({
    'category': 'RMS Features',
    'metric': 'Max Abnormal/Normal Ratio',
    'value': f"{rms_ratios_df['abnormal_to_normal_ratio'].max():.3f}"
})
summary_table.append({
    'category': 'RMS Features',
    'metric': 'High Correlation Pairs (|r|>0.95)',
    'value': len(high_corr_rms) if len(high_corr_rms) > 0 else 0
})

# Top discriminative RMS
top_3_rms = rms_ratios_df.head(3)
for idx, row in top_3_rms.iterrows():
    summary_table.append({
        'category': 'Top RMS Features',
        'metric': row['feature'],
        'value': f"Ratio={row['abnormal_to_normal_ratio']:.3f} (+{row['pct_increase']:.1f}%)"
    })

summary_df = pd.DataFrame(summary_table)
summary_df.to_csv(OUTPUT_PATH / 'visualization_summary.csv', index=False)
print(f"  ✓ Saved: visualization_summary.csv")
print()

# ============================================================================
# Final Report
# ============================================================================
print("=" * 80)
print("시각화 완료!")
print("=" * 80)
print()
print("생성된 파일:")
print()
print("📊 분석 데이터:")
print(f"  1. rms_statistics_by_class.csv - RMS 클래스별 통계")
print(f"  2. rms_abnormal_to_normal_ratios.csv - RMS 불량/정상 비율")
print(f"  3. rms_correlation_matrix.csv - RMS 상관계수 행렬")
if len(high_corr_rms) > 0:
    print(f"  4. rms_high_correlation_pairs.csv - 고상관 RMS 쌍")
print(f"  5. visualization_summary.csv - 시각화 종합 요약")
print()
print("📈 시각화 + 데이터:")
print(f"  - {len(scatter_pairs)} scatter plots + CSV data")
print(f"  - {len(top_rms)} boxplots + CSV data + summary stats")
print()
print(f"저장 위치: {VIZ_PATH}")
print()
print("=" * 80)
print("주요 발견사항:")
print("=" * 80)

print(f"\n🔴 RMS 불량/정상 비율 Top 3:")
for idx, row in rms_ratios_df.head(3).iterrows():
    print(f"  {row['feature']}: {row['abnormal_to_normal_ratio']:.3f}x "
          f"(+{row['pct_increase']:.1f}%)")

print(f"\n📊 RMS 고상관 쌍: {len(high_corr_rms) if len(high_corr_rms) > 0 else 0}개")
if len(high_corr_rms) > 0:
    print("  Top 3:")
    for idx, row in high_corr_rms.head(3).iterrows():
        print(f"    {row['feature1']} ↔ {row['feature2']}: r={row['correlation']:.4f}")

print("\n✅ 모든 시각화 데이터가 CSV로 저장되어 AI 분석 가능합니다!")
