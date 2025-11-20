"""
Main pipeline script for data processing and feature extraction
"""
import logging
from pathlib import Path

import pandas as pd

# Add src to path
import sys
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from src.utils.helpers import load_config, setup_logging, resolve_path
from src.io.load_csv import load_all_csv_files, create_metadata_table
from src.io.load_labels import load_labels_from_excel, create_file_master_table
from src.preprocess.clean import clean_timeseries
from src.preprocess.quality import create_quality_report
from src.preprocess.split_strategy import assign_split_sets
from src.preprocess.segment import create_windows_metadata
from src.preprocess.balance import balance_train_windows, calculate_class_distribution
from src.features.time_domain import extract_features_from_windows


def main():
    """
    Run complete data processing pipeline
    """
    # ========================================================================
    # 1. Setup
    # ========================================================================
    print("="*80)
    print("Actuator Vibration Analysis - Data Processing Pipeline")
    print("="*80)

    # Load configurations
    paths_config = load_config('paths')
    params_config = load_config('params_eda')

    # Setup logging
    log_config = params_config.get('logging', {})
    log_file = resolve_path(log_config.get('log_file', 'data/interim/pipeline.log'))

    setup_logging(
        log_level=log_config.get('level', 'INFO'),
        log_file=log_file
    )

    logger = logging.getLogger(__name__)
    logger.info("Starting pipeline")

    # Get project root
    project_root = Path(__file__).parent

    # ========================================================================
    # 2. Load raw data
    # ========================================================================
    print("\n[Step 1/9] Loading CSV files...")

    loaded_100w = load_all_csv_files(
        project_root / paths_config['raw']['csv_100w'],
        product='100W'
    )

    loaded_200w = load_all_csv_files(
        project_root / paths_config['raw']['csv_200w'],
        product='200W'
    )

    loaded_files = loaded_100w + loaded_200w

    logger.info(f"Total files loaded: {len(loaded_files)}")
    print(f"  ✓ Loaded {len(loaded_100w)} files (100W) + {len(loaded_200w)} files (200W)")

    # ========================================================================
    # 3. Create metadata table
    # ========================================================================
    print("\n[Step 2/9] Creating metadata table...")

    metadata_df = create_metadata_table(loaded_files)

    print(f"  ✓ Metadata table: {len(metadata_df)} files")

    # ========================================================================
    # 4. Load labels and create file master table
    # ========================================================================
    print("\n[Step 3/9] Loading labels from Excel...")

    excel_path = project_root / paths_config['raw']['excel_labels']
    labels_df = load_labels_from_excel(excel_path)

    print(f"  ✓ Loaded {len(labels_df)} label records")

    print("\n[Step 4/9] Creating file master table...")

    file_master_df = create_file_master_table(metadata_df, labels_df, params_config)

    print(f"  ✓ File master table: {len(file_master_df)} files")
    print(f"     Normal samples: {file_master_df['is_normal'].sum()}")
    print(f"     Label distribution:")
    for label, count in file_master_df['label_raw'].value_counts().items():
        print(f"       {label}: {count}")

    # Save file master table
    file_master_path = resolve_path(paths_config['interim']['file_master'])
    file_master_path.parent.mkdir(parents=True, exist_ok=True)
    file_master_df.to_parquet(file_master_path, index=False)
    print(f"  ✓ Saved to {file_master_path}")

    # ========================================================================
    # 5. Data cleaning and quality checks
    # ========================================================================
    print("\n[Step 5/9] Cleaning timeseries data...")

    quality_params = params_config.get('quality', {})

    for item in loaded_files:
        ts_df = item['timeseries']

        # Clean timeseries
        cleaned_ts, quality_info = clean_timeseries(ts_df, quality_params)

        # Update timeseries
        item['timeseries'] = cleaned_ts
        item['quality_info'] = quality_info

    print("  ✓ All timeseries cleaned")

    # Create quality report
    print("\n[Step 6/9] Creating quality report...")

    quality_df = create_quality_report(file_master_df, loaded_files, quality_params)

    quality_path = resolve_path(paths_config['interim']['quality_report'])
    quality_df.to_csv(quality_path, index=False)

    print(f"  ✓ Quality report: {quality_df['is_usable'].sum()}/{len(quality_df)} files usable")
    print(f"  ✓ Saved to {quality_path}")

    # ========================================================================
    # 6. Assign train/val/test splits
    # ========================================================================
    print("\n[Step 7/9] Assigning train/val/test splits...")

    split_mapping_path = resolve_path(paths_config['interim']['split_mapping'])

    file_master_df = assign_split_sets(
        file_master_df,
        params_config,
        output_path=split_mapping_path
    )

    # Update file master table with splits
    file_master_df.to_parquet(file_master_path, index=False)

    print("  ✓ Split assignments:")
    print(f"     time_split (normal): {(file_master_df['split_set'] == 'time_split').sum()}")
    print(f"     train (abnormal): {(file_master_df['split_set'] == 'train').sum()}")
    print(f"     val (abnormal): {(file_master_df['split_set'] == 'val').sum()}")
    print(f"     test (abnormal): {(file_master_df['split_set'] == 'test').sum()}")

    # ========================================================================
    # 7. Generate window segments
    # ========================================================================
    print("\n[Step 8/9] Generating window segments...")

    windows_df = create_windows_metadata(loaded_files, file_master_df, params_config)

    windows_path = resolve_path(paths_config['interim']['windows_meta'])
    windows_df.to_parquet(windows_path, index=False)

    print(f"  ✓ Generated {len(windows_df)} windows")
    print(f"  ✓ Saved to {windows_path}")

    # Calculate distribution before balancing
    dist_before = calculate_class_distribution(windows_df)
    print("\n  Distribution before balancing:")
    for split_set, stats in dist_before.items():
        if stats['total'] > 0:
            ratio_str = f"{stats['ratio']:.2f}" if stats['ratio'] is not None else "N/A"
            print(f"    {split_set}: normal={stats['normal']}, abnormal={stats['abnormal']}, ratio={ratio_str}")

    # ========================================================================
    # 8. Balance train set
    # ========================================================================
    print("\n[Step 9/9] Balancing train set...")

    balanced_df = balance_train_windows(windows_df, params_config)

    balanced_path = resolve_path(paths_config['processed']['windows_balanced'])
    balanced_path.parent.mkdir(parents=True, exist_ok=True)
    balanced_df.to_parquet(balanced_path, index=False)

    print(f"  ✓ Saved to {balanced_path}")

    # Calculate distribution after balancing
    dist_after = calculate_class_distribution(balanced_df)
    print("\n  Distribution after balancing:")
    for split_set, stats in dist_after.items():
        if stats['total'] > 0:
            ratio_str = f"{stats['ratio']:.2f}" if stats['ratio'] is not None else "N/A"
            print(f"    {split_set}: normal={stats['normal']}, abnormal={stats['abnormal']}, ratio={ratio_str}")

    # ========================================================================
    # 9. Extract features
    # ========================================================================
    print("\n[Bonus] Extracting time-domain features...")

    features_df = extract_features_from_windows(
        loaded_files,
        balanced_df,
        params_config
    )

    # Save combined features
    features_combined_path = resolve_path(paths_config['processed']['features_combined'])
    features_df.to_parquet(features_combined_path, index=False)

    print(f"  ✓ Features extracted: {len(features_df)} windows × {len(features_df.columns)} features")
    print(f"  ✓ Saved to {features_combined_path}")

    # Save product-specific features
    for product in ['100W', '200W']:
        product_features = features_df[features_df['product'] == product].copy()

        if len(product_features) > 0:
            product_path_key = f'features_{product.lower()}'
            product_path = resolve_path(paths_config['processed'][product_path_key])
            product_features.to_parquet(product_path, index=False)

            print(f"  ✓ {product}: {len(product_features)} windows saved to {product_path}")

    # ========================================================================
    # Done
    # ========================================================================
    print("\n" + "="*80)
    print("Pipeline completed successfully!")
    print("="*80)

    print("\nNext steps:")
    print("  1. Review quality report: data/interim/quality_report.csv")
    print("  2. Explore data with EDA notebooks in notebooks/eda/")
    print("  3. Check feature distributions and class separability")

    logger.info("Pipeline completed successfully")


if __name__ == '__main__':
    main()
