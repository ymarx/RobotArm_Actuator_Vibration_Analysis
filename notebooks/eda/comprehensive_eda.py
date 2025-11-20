"""
Comprehensive EDA Script
종합 탐색적 데이터 분석

참조 문서:
- 01_단계별분석전략.md
- 03_eda_and_preprocessing_plan.md

분석 항목:
1. 데이터 전처리 검증
2. Feature 분포 분석 (정상 vs 불량)
3. 제품별/방향별 특성 비교
4. 클래스 분리 가능성 평가
5. 2차 단계 준비사항 정리
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats

# Paths
BASE_PATH = Path(__file__).parent.parent.parent
DATA_PATH = BASE_PATH / "data"
OUTPUT_PATH = BASE_PATH / "claudedocs" / "eda_results"
OUTPUT_PATH.mkdir(exist_ok=True, parents=True)

print("=" * 80)
print("종합 탐색적 데이터 분석 (Comprehensive EDA)")
print("=" * 80)
print()

# ============================================================================
# 1. 데이터 로딩
# ============================================================================
print("[1/8] 데이터 로딩...")

file_master = pd.read_parquet(DATA_PATH / "interim" / "file_master_v1.parquet")
quality_report = pd.read_csv(DATA_PATH / "interim" / "quality_report.csv")
windows_balanced = pd.read_parquet(DATA_PATH / "processed" / "windows_balanced_v1.parquet")
features = pd.read_parquet(DATA_PATH / "processed" / "features_combined_v1.parquet")

print(f"  ✓ File master: {len(file_master)} files")
print(f"  ✓ Quality report: {len(quality_report)} files")
print(f"  ✓ Windows (balanced): {len(windows_balanced)}")
print(f"  ✓ Features: {len(features)} windows × {len(features.columns)} columns")
print()

# ============================================================================
# 2. 데이터 전처리 검증
# ============================================================================
print("[2/8] 데이터 전처리 검증...")

# 2.1 품질 통과율
usable_count = quality_report['is_usable'].sum()
total_count = len(quality_report)
usable_pct = usable_count / total_count * 100

print(f"\n품질 검사 결과:")
print(f"  ✓ 총 파일: {total_count}")
print(f"  ✓ 사용 가능: {usable_count} ({usable_pct:.1f}%)")
print(f"  ✓ 문제 있음: {total_count - usable_count}")

# 제품별 품질
for product in ['100W', '200W']:
    product_qr = quality_report[quality_report['file_id'].str.startswith(product)]
    product_usable = product_qr['is_usable'].sum()
    print(f"  - {product}: {product_usable}/{len(product_qr)} usable")

# 2.2 라벨 분포
print(f"\n라벨 분포:")
print(file_master['label_raw'].value_counts())
print(f"\n이진 라벨 분포:")
print(file_master['label_binary'].value_counts())
print(f"  - 정상 (1): {(file_master['label_binary'] == 1).sum()}")
print(f"  - 불량 (0): {(file_master['label_binary'] == 0).sum()}")

# 2.3 윈도우 분포
print(f"\nSplit 분포:")
for split in ['train', 'val', 'test']:
    split_windows = windows_balanced[windows_balanced['split_set'] == split]
    normal = (split_windows['label_binary'] == 1).sum()
    abnormal = (split_windows['label_binary'] == 0).sum()
    ratio = normal / abnormal if abnormal > 0 else 0
    print(f"  {split:5s}: 정상={normal:3d}, 불량={abnormal:3d}, 비율={ratio:.2f}")

# 2.4 Missing values 확인
numeric_cols = features.select_dtypes(include=[np.number]).columns
missing = features[numeric_cols].isnull().sum()
missing_features = missing[missing > 0]

if len(missing_features) > 0:
    print(f"\n⚠️  Missing values 발견:")
    print(f"  {len(missing_features)}개 feature에서 결측치 존재")
    print(f"  총 결측 윈도우 수: {missing_features.max()}")
else:
    print("\n✓ Missing values 없음")

print()

# ============================================================================
# 3. Feature 기본 통계
# ============================================================================
print("[3/8] Feature 기본 통계...")

# 3.1 Feature 분류
sensor_features = [col for col in features.columns if any(
    sensor in col for sensor in ['acc_X', 'acc_Y', 'acc_Z', 'Gyro_X', 'Gyro_Y', 'Gyro_Z', 'acc_Sum']
) and any(stat in col for stat in ['mean', 'std', 'rms', 'peak', 'crest', 'kurtosis', 'skewness'])]

meta_features = ['product_100w', 'direction_cw']

print(f"  ✓ 센서 features: {len(sensor_features)}")
print(f"  ✓ 메타 features: {len(meta_features)}")

# 3.2 기본 통계량 계산
stats_summary = features[sensor_features].describe()
stats_summary.to_csv(OUTPUT_PATH / "feature_statistics.csv")
print(f"  ✓ Saved: feature_statistics.csv")

# 3.3 Feature별 변동계수 (CV = std/mean)
cv_values = {}
for col in sensor_features:
    mean_val = features[col].mean()
    std_val = features[col].std()
    if abs(mean_val) > 1e-10:
        cv_values[col] = std_val / abs(mean_val)
    else:
        cv_values[col] = np.nan

cv_df = pd.DataFrame.from_dict(cv_values, orient='index', columns=['CV'])
cv_df = cv_df.sort_values('CV', ascending=False)
cv_df.to_csv(OUTPUT_PATH / "feature_cv.csv")
print(f"  ✓ Saved: feature_cv.csv (변동계수)")

print()

# ============================================================================
# 4. 클래스별 Feature 분포 비교
# ============================================================================
print("[4/8] 클래스별 Feature 분포 비교...")

# Features already contains label_binary and split_set
df_analysis = features.copy()

# Train set만 분석 (balanced된 데이터)
train_df = df_analysis[df_analysis['split_set'] == 'train']
normal_df = train_df[train_df['label_binary'] == 1]
abnormal_df = train_df[train_df['label_binary'] == 0]

print(f"  Train set 분석: 정상={len(normal_df)}, 불량={len(abnormal_df)}")

# 4.1 각 feature별 정상 vs 불량 통계
comparison_list = []
for col in sensor_features:
    if col not in train_df.columns:
        continue

    normal_vals = normal_df[col].dropna()
    abnormal_vals = abnormal_df[col].dropna()

    if len(normal_vals) == 0 or len(abnormal_vals) == 0:
        continue

    # T-test
    t_stat, p_value = stats.ttest_ind(normal_vals, abnormal_vals, equal_var=False)

    # Effect size (Cohen's d)
    pooled_std = np.sqrt((normal_vals.std()**2 + abnormal_vals.std()**2) / 2)
    cohens_d = (normal_vals.mean() - abnormal_vals.mean()) / pooled_std if pooled_std > 0 else 0

    comparison_list.append({
        'feature': col,
        'normal_mean': normal_vals.mean(),
        'normal_std': normal_vals.std(),
        'abnormal_mean': abnormal_vals.mean(),
        'abnormal_std': abnormal_vals.std(),
        'mean_diff': normal_vals.mean() - abnormal_vals.mean(),
        'mean_diff_pct': ((normal_vals.mean() - abnormal_vals.mean()) / abnormal_vals.mean() * 100) if abnormal_vals.mean() != 0 else 0,
        't_statistic': t_stat,
        'p_value': p_value,
        'cohens_d': cohens_d,
        'significant': p_value < 0.05,
        'large_effect': abs(cohens_d) > 0.8
    })

comparison_df = pd.DataFrame(comparison_list)
comparison_df = comparison_df.sort_values('p_value')
comparison_df.to_csv(OUTPUT_PATH / "class_comparison.csv", index=False)
print(f"  ✓ Saved: class_comparison.csv")

# 통계적으로 유의미한 feature 개수
significant_count = comparison_df['significant'].sum()
large_effect_count = comparison_df['large_effect'].sum()
print(f"  ✓ 통계적으로 유의미한 features: {significant_count}/{len(comparison_df)} (p<0.05)")
print(f"  ✓ Large effect size features: {large_effect_count}/{len(comparison_df)} (|d|>0.8)")

# Top 10 discriminative features
print(f"\n  Top 10 가장 구별력 높은 features (Cohen's d 기준):")
top10 = comparison_df.nlargest(10, 'cohens_d', keep='all')
for idx, row in top10.iterrows():
    print(f"    {row['feature']:30s}: d={row['cohens_d']:6.3f}, p={row['p_value']:.2e}")

print()

# ============================================================================
# 5. 제품별/방향별 특성 비교
# ============================================================================
print("[5/8] 제품별/방향별 특성 비교...")

# 5.1 제품별 (100W vs 200W)
product_comparison = []
for col in sensor_features[:10]:  # 대표 10개만
    if col not in df_analysis.columns:
        continue

    w100 = df_analysis[df_analysis['product_100w'] == 1][col].dropna()
    w200 = df_analysis[df_analysis['product_100w'] == 0][col].dropna()

    if len(w100) > 0 and len(w200) > 0:
        t_stat, p_value = stats.ttest_ind(w100, w200, equal_var=False)
        product_comparison.append({
            'feature': col,
            '100W_mean': w100.mean(),
            '200W_mean': w200.mean(),
            'diff_pct': ((w100.mean() - w200.mean()) / w200.mean() * 100) if w200.mean() != 0 else 0,
            'p_value': p_value
        })

product_comp_df = pd.DataFrame(product_comparison)
product_comp_df.to_csv(OUTPUT_PATH / "product_comparison.csv", index=False)
print(f"  ✓ Saved: product_comparison.csv (100W vs 200W)")

# 5.2 방향별 (CW vs CCW)
direction_comparison = []
for col in sensor_features[:10]:  # 대표 10개만
    if col not in df_analysis.columns:
        continue

    cw = df_analysis[df_analysis['direction_cw'] == 1][col].dropna()
    ccw = df_analysis[df_analysis['direction_cw'] == 0][col].dropna()

    if len(cw) > 0 and len(ccw) > 0:
        t_stat, p_value = stats.ttest_ind(cw, ccw, equal_var=False)
        direction_comparison.append({
            'feature': col,
            'CW_mean': cw.mean(),
            'CCW_mean': ccw.mean(),
            'diff_pct': ((cw.mean() - ccw.mean()) / ccw.mean() * 100) if ccw.mean() != 0 else 0,
            'p_value': p_value
        })

direction_comp_df = pd.DataFrame(direction_comparison)
direction_comp_df.to_csv(OUTPUT_PATH / "direction_comparison.csv", index=False)
print(f"  ✓ Saved: direction_comparison.csv (CW vs CCW)")

print()

# ============================================================================
# 6. Correlation 분석
# ============================================================================
print("[6/8] Correlation 분석...")

# Train set의 sensor features만 상관계수 계산
corr_features = sensor_features[:20]  # 너무 많으면 계산 오래 걸림
corr_matrix = train_df[corr_features].corr()
corr_matrix.to_csv(OUTPUT_PATH / "feature_correlation.csv")
print(f"  ✓ Saved: feature_correlation.csv")

# 고상관 feature 쌍 찾기 (|r| > 0.9)
high_corr_pairs = []
for i in range(len(corr_matrix.columns)):
    for j in range(i+1, len(corr_matrix.columns)):
        corr_val = corr_matrix.iloc[i, j]
        if abs(corr_val) > 0.9:
            high_corr_pairs.append({
                'feature1': corr_matrix.columns[i],
                'feature2': corr_matrix.columns[j],
                'correlation': corr_val
            })

if high_corr_pairs:
    high_corr_df = pd.DataFrame(high_corr_pairs)
    high_corr_df.to_csv(OUTPUT_PATH / "high_correlation_pairs.csv", index=False)
    print(f"  ✓ 고상관 feature 쌍 (|r|>0.9): {len(high_corr_pairs)}개")
else:
    print(f"  ✓ 고상관 feature 쌍 없음")

print()

# ============================================================================
# 7. Outlier 분석
# ============================================================================
print("[7/8] Outlier 분석...")

outlier_summary = []
for col in sensor_features:
    if col not in train_df.columns:
        continue

    vals = train_df[col].dropna()
    if len(vals) == 0:
        continue

    Q1 = vals.quantile(0.25)
    Q3 = vals.quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 3 * IQR
    upper_bound = Q3 + 3 * IQR

    outliers = ((vals < lower_bound) | (vals > upper_bound)).sum()
    outlier_pct = outliers / len(vals) * 100

    if outliers > 0:
        outlier_summary.append({
            'feature': col,
            'outlier_count': outliers,
            'outlier_pct': outlier_pct,
            'Q1': Q1,
            'Q3': Q3,
            'IQR': IQR,
            'lower_bound': lower_bound,
            'upper_bound': upper_bound,
            'min': vals.min(),
            'max': vals.max()
        })

if outlier_summary:
    outlier_df = pd.DataFrame(outlier_summary).sort_values('outlier_pct', ascending=False)
    outlier_df.to_csv(OUTPUT_PATH / "outlier_analysis.csv", index=False)
    print(f"  ✓ Outlier가 있는 features: {len(outlier_df)}/{len(sensor_features)}")
    print(f"  ✓ Top outlier features:")
    for idx, row in outlier_df.head(5).iterrows():
        print(f"    {row['feature']:30s}: {row['outlier_pct']:.1f}% outliers")
else:
    print(f"  ✓ Outlier 없음")

print()

# ============================================================================
# 8. 종합 요약 및 2차 단계 권장사항
# ============================================================================
print("[8/8] 종합 요약 생성...")

summary = {
    'total_files': len(file_master),
    'usable_files': usable_count,
    'usable_pct': usable_pct,
    'total_windows': len(windows_balanced),
    'train_windows': len(train_df),
    'val_windows': len(df_analysis[df_analysis['split_set'] == 'val']),
    'test_windows': len(df_analysis[df_analysis['split_set'] == 'test']),
    'normal_files': (file_master['label_binary'] == 1).sum(),
    'abnormal_files': (file_master['label_binary'] == 0).sum(),
    'total_features': len(sensor_features),
    'significant_features': significant_count,
    'large_effect_features': large_effect_count,
    'high_corr_pairs': len(high_corr_pairs) if high_corr_pairs else 0,
    'features_with_outliers': len(outlier_df) if outlier_summary else 0,
    'missing_value_features': len(missing_features) if len(missing_features) > 0 else 0
}

summary_df = pd.DataFrame([summary]).T
summary_df.columns = ['Value']
summary_df.to_csv(OUTPUT_PATH / "eda_summary.csv")
print(f"  ✓ Saved: eda_summary.csv")

print()
print("=" * 80)
print("EDA 완료!")
print("=" * 80)
print()
print(f"결과 저장 위치: {OUTPUT_PATH}/")
print()
print("생성된 파일:")
print("  1. feature_statistics.csv - Feature 기본 통계량")
print("  2. feature_cv.csv - Feature 변동계수")
print("  3. class_comparison.csv - 정상 vs 불량 비교")
print("  4. product_comparison.csv - 100W vs 200W 비교")
print("  5. direction_comparison.csv - CW vs CCW 비교")
print("  6. feature_correlation.csv - Feature 상관계수")
if high_corr_pairs:
    print("  7. high_correlation_pairs.csv - 고상관 feature 쌍")
if outlier_summary:
    print("  8. outlier_analysis.csv - Outlier 분석")
print("  9. eda_summary.csv - 종합 요약")
print()

# 2차 단계 권장사항
print("=" * 80)
print("2차 단계 준비사항")
print("=" * 80)
print()
print(f"✓ 클래스 분리 가능성: {'양호' if significant_count > len(sensor_features) * 0.3 else '보통'}")
print(f"  - 통계적으로 유의미한 features: {significant_count}/{len(sensor_features)}")
print(f"  - Large effect size features: {large_effect_count}")
print()
print("권장 다음 단계:")
print("  1. Missing values 처리 (5개 윈도우 제거 또는 imputation)")
print("  2. XGBoost 베이스라인 모델 학습")
print("  3. Feature importance 분석")
print("  4. 주파수 domain features 추가 검토")
print("  5. Autoencoder anomaly detection 구현")
print()
