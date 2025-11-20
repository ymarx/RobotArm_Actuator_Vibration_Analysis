"""
01_basic_statistics.py

기본 통계 및 데이터 분포 분석
- Feature 통계량
- 클래스별 분포
- Missing value 확인
- Outlier 탐지
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Note: Skipping matplotlib/seaborn due to version conflicts
# Will generate CSV outputs for analysis

# Paths
BASE_PATH = Path(__file__).parent.parent.parent
DATA_PATH = BASE_PATH / "data" / "processed"
OUTPUT_PATH = BASE_PATH / "claudedocs" / "eda_results"
OUTPUT_PATH.mkdir(exist_ok=True, parents=True)

print("=" * 80)
print("탐색적 데이터 분석 (EDA) - 01. 기본 통계")
print("=" * 80)
print()

# ============================================================================
# 1. 데이터 로딩
# ============================================================================
print("[1/6] 데이터 로딩...")

features_df = pd.read_parquet(DATA_PATH / "features_combined_v1.parquet")
windows_df = pd.read_parquet(DATA_PATH / "windows_balanced_v1.parquet")

# Merge to get labels
df = features_df.merge(
    windows_df[['window_id', 'split_set', 'label_binary', 'label_raw']],
    on='window_id',
    how='left'
)

print(f"  ✓ Total windows: {len(df)}")
print(f"  ✓ Features: {len(df.columns)}")
print()

# ============================================================================
# 2. Feature 분류
# ============================================================================
print("[2/6] Feature 분류...")

# Meta columns
meta_cols = ['window_id', 'split_set', 'label_binary', 'label_raw', 'sample', 'label_weight',
             'file_id', 'product', 'direction', 'window_duration', 'window_num_samples']
meta_features = ['product_100w', 'direction_cw']

# Get all numeric features (exclude meta columns and meta features)
all_cols = df.columns.tolist()
exclude_cols = meta_cols + meta_features
numeric_features = [col for col in all_cols if col not in exclude_cols]

# Filter only actual numeric columns
numeric_features = [col for col in numeric_features if df[col].dtype in ['float64', 'int64']]

print(f"  ✓ Numeric features: {len(numeric_features)}")
print(f"  ✓ Meta features: {len(meta_features)}")
print()

# ============================================================================
# 3. 기본 통계량
# ============================================================================
print("[3/6] 기본 통계량 계산...")

stats_summary = df[numeric_features].describe()
stats_summary.to_csv(OUTPUT_PATH / "basic_statistics.csv")

print("  ✓ 기본 통계량:")
print(stats_summary.iloc[:, :5].round(3))
print(f"  ✓ Saved to: {OUTPUT_PATH / 'basic_statistics.csv'}")
print()

# ============================================================================
# 4. Missing Value 확인
# ============================================================================
print("[4/6] Missing Value 확인...")

missing_counts = df[numeric_features].isnull().sum()
missing_features = missing_counts[missing_counts > 0]

if len(missing_features) > 0:
    print(f"  ⚠️  Missing values found in {len(missing_features)} features:")
    print(missing_features)
else:
    print("  ✓ No missing values")
print()

# ============================================================================
# 5. 클래스별 통계
# ============================================================================
print("[5/6] 클래스별 통계...")

class_stats = []
for split in ['train', 'val', 'test']:
    split_df = df[df['split_set'] == split]

    for label in [0, 1]:  # 0: abnormal, 1: normal
        label_df = split_df[split_df['label_binary'] == label]

        if len(label_df) > 0:
            label_name = '정상' if label == 1 else '불량'

            # Calculate statistics for first 5 features as example
            for col in numeric_features[:5]:
                class_stats.append({
                    'split': split,
                    'label': label_name,
                    'feature': col,
                    'count': len(label_df),
                    'mean': label_df[col].mean(),
                    'std': label_df[col].std(),
                    'min': label_df[col].min(),
                    'max': label_df[col].max()
                })

class_stats_df = pd.DataFrame(class_stats)
class_stats_df.to_csv(OUTPUT_PATH / "class_statistics.csv", index=False)

print("  ✓ 클래스별 통계:")
print(class_stats_df.head(10))
print(f"  ✓ Saved to: {OUTPUT_PATH / 'class_statistics.csv'}")
print()

# ============================================================================
# 6. Outlier 탐지 (IQR 방식)
# ============================================================================
print("[6/6] Outlier 탐지...")

outlier_summary = []

for col in numeric_features:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 3 * IQR  # 3 IQR (extreme outliers)
    upper_bound = Q3 + 3 * IQR

    outliers = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
    outlier_pct = outliers / len(df) * 100

    if outliers > 0:
        outlier_summary.append({
            'feature': col,
            'outlier_count': outliers,
            'outlier_pct': outlier_pct,
            'lower_bound': lower_bound,
            'upper_bound': upper_bound,
            'min': df[col].min(),
            'max': df[col].max()
        })

outlier_df = pd.DataFrame(outlier_summary).sort_values('outlier_pct', ascending=False)
outlier_df.to_csv(OUTPUT_PATH / "outlier_summary.csv", index=False)

print(f"  ✓ Features with outliers: {len(outlier_df)}/{len(numeric_features)}")
if len(outlier_df) > 0:
    print("  Top 5 features with most outliers:")
    print(outlier_df.head())
print(f"  ✓ Saved to: {OUTPUT_PATH / 'outlier_summary.csv'}")
print()

# ============================================================================
# 7. Summary Report
# ============================================================================
print("=" * 80)
print("Summary")
print("=" * 80)
print(f"✓ Total windows: {len(df)}")
print(f"✓ Numeric features: {len(numeric_features)}")
print(f"✓ Missing values: {'None' if len(missing_features) == 0 else len(missing_features)}")
print(f"✓ Features with outliers: {len(outlier_df)}/{len(numeric_features)}")
print()
print(f"Results saved to: {OUTPUT_PATH}/")
print("  - basic_statistics.csv")
print("  - class_statistics.csv")
print("  - outlier_summary.csv")
print()
