"""
ECMiner Stage 1: Complete Preprocessing Pipeline
Raw CSV → 18 Features (9 Basic + 9 Band RMS) → CSV Output

이 스크립트는 Phase3-1 전까지의 모든 전처리를 수행합니다:
1. Raw CSV 로드 (data/100W, data/200W)
2. 레이블 매칭
3. 데이터 품질 검사
4. Train/Val/Test 분할
5. 윈도우 세그먼트 생성
6. 기본 특성 추출 (9개)
7. Band RMS 특성 추출 (9개)
8. ECMiner 입력용 CSV 생성 (18 features)
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from scipy import signal
import logging

# Setup
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT / 'src'))

from src.utils.helpers import load_config
from src.io.load_csv import load_all_csv_files, parse_csv_with_metadata
from src.io.load_labels import load_labels_from_excel, create_file_master_table
from src.preprocess.clean import clean_timeseries
from src.preprocess.quality import create_quality_report
from src.preprocess.split_strategy import assign_split_sets
from src.preprocess.segment import create_windows_metadata

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

# ============================================================================
# Configuration
# ============================================================================

OUTPUT_CSV = PROJECT_ROOT / "ecminer_stage1_output.csv"

# Phase2 Step 3-4 18 features
BASIC_FEATURES = [
    'acc_Y_rms', 'acc_Y_peak', 'acc_Y_crest',
    'acc_Sum_rms', 'acc_Sum_peak', 'acc_Sum_crest',
    'Gyro_Y_rms', 'Gyro_Y_peak', 'Gyro_Y_crest'
]

BAND_RMS_FEATURES = [
    'acc_Y_rms_low', 'acc_Y_rms_mid', 'acc_Y_rms_high',
    'acc_Sum_rms_low', 'acc_Sum_rms_mid', 'acc_Sum_rms_high',
    'Gyro_Y_rms_low', 'Gyro_Y_rms_mid', 'Gyro_Y_rms_high'
]

ALL_FEATURES = BASIC_FEATURES + BAND_RMS_FEATURES

# Band definitions (Hz)
BANDS = {
    'low': (1, 10),      # Structural vibration, unbalance
    'mid': (10, 50),     # Rotation harmonics
    'high': (50, 150)    # Bearing defects, impacts
}

SAMPLING_RATE = 512.0  # Hz


# ============================================================================
# Feature Extraction Functions
# ============================================================================

def extract_basic_features(signal_data: np.ndarray, channel: str) -> dict:
    """Extract 9 basic time-domain features"""
    features = {}

    if len(signal_data) == 0:
        return features

    # RMS
    rms = np.sqrt(np.mean(signal_data ** 2))
    features[f'{channel}_rms'] = rms

    # Peak (max absolute value)
    peak = np.max(np.abs(signal_data))
    features[f'{channel}_peak'] = peak

    # Crest Factor
    if rms > 0:
        features[f'{channel}_crest'] = peak / rms
    else:
        features[f'{channel}_crest'] = 0.0

    return features


def compute_band_rms(signal_data: np.ndarray, fs: float, band: tuple) -> float:
    """
    Compute RMS of signal in specific frequency band

    Args:
        signal_data: Time-domain signal
        fs: Sampling frequency
        band: (low_freq, high_freq) in Hz

    Returns:
        RMS value of filtered signal
    """
    if len(signal_data) < 100:
        return np.nan

    try:
        # Butterworth bandpass filter (4th order)
        nyquist = fs / 2
        low_norm = band[0] / nyquist
        high_norm = band[1] / nyquist

        # Ensure valid frequency range
        low_norm = max(0.01, min(0.99, low_norm))
        high_norm = max(0.01, min(0.99, high_norm))

        if low_norm >= high_norm:
            return np.nan

        sos = signal.butter(4, [low_norm, high_norm], btype='band', output='sos')
        filtered = signal.sosfilt(sos, signal_data)

        # Compute RMS
        rms = np.sqrt(np.mean(filtered ** 2))
        return rms

    except Exception as e:
        logger.debug(f"Band RMS error: {e}")
        return np.nan


def extract_band_rms_features(signal_data: np.ndarray, channel: str, fs: float) -> dict:
    """Extract 3 band RMS features per channel"""
    features = {}

    for band_name, band_range in BANDS.items():
        band_rms = compute_band_rms(signal_data, fs, band_range)
        feature_name = f"{channel}_rms_{band_name}"
        features[feature_name] = band_rms

    return features


def extract_window_features(window_data: pd.DataFrame, fs: float) -> dict:
    """Extract all 18 features from a window"""
    features = {}

    # Channels to process
    channels = ['acc_Y', 'acc_Sum', 'Gyro_Y']

    for channel in channels:
        col_name = channel.replace('_', '-')  # acc_Y → acc-Y

        if col_name not in window_data.columns:
            # Fill with NaN if channel missing
            for feat in BASIC_FEATURES:
                if feat.startswith(channel):
                    features[feat] = np.nan
            for feat in BAND_RMS_FEATURES:
                if feat.startswith(channel):
                    features[feat] = np.nan
            continue

        signal_data = window_data[col_name].values

        # Basic features
        basic_feats = extract_basic_features(signal_data, channel)
        features.update(basic_feats)

        # Band RMS features
        band_feats = extract_band_rms_features(signal_data, channel, fs)
        features.update(band_feats)

    return features


# ============================================================================
# Main Pipeline
# ============================================================================

def main():
    """Complete preprocessing pipeline for ECMiner"""

    print("=" * 80)
    print("ECMiner Stage 1: Complete Preprocessing Pipeline")
    print("Raw CSV → 18 Features → ECMiner CSV")
    print("=" * 80)

    # ------------------------------------------------------------------------
    # 1. Load Configuration
    # ------------------------------------------------------------------------
    print("\n[Step 1/8] Loading configuration...")
    paths_config = load_config('paths')
    params_config = load_config('params_eda')

    # ------------------------------------------------------------------------
    # 2. Load Raw CSV Files
    # ------------------------------------------------------------------------
    print("\n[Step 2/8] Loading raw CSV files...")

    loaded_100w = load_all_csv_files(
        PROJECT_ROOT / paths_config['raw']['csv_100w'],
        product='100W'
    )

    loaded_200w = load_all_csv_files(
        PROJECT_ROOT / paths_config['raw']['csv_200w'],
        product='200W'
    )

    loaded_files = loaded_100w + loaded_200w
    logger.info(f"  ✓ Loaded {len(loaded_files)} files (100W: {len(loaded_100w)}, 200W: {len(loaded_200w)})")

    # ------------------------------------------------------------------------
    # 3. Load Labels and Create File Master
    # ------------------------------------------------------------------------
    print("\n[Step 3/8] Loading labels and creating file master...")

    excel_path = PROJECT_ROOT / paths_config['raw']['excel_labels']
    labels_df = load_labels_from_excel(excel_path)

    from src.io.load_csv import create_metadata_table
    metadata_df = create_metadata_table(loaded_files)

    file_master_df = create_file_master_table(metadata_df, labels_df, params_config)
    logger.info(f"  ✓ File master: {len(file_master_df)} files, Normal: {file_master_df['is_normal'].sum()}")

    # ------------------------------------------------------------------------
    # 4. Quality Check and Clean
    # ------------------------------------------------------------------------
    print("\n[Step 4/8] Quality check and data cleaning...")

    quality_params = params_config.get('quality', {})

    for item in loaded_files:
        ts_df = item['timeseries']
        cleaned_ts, quality_info = clean_timeseries(ts_df, quality_params)
        item['timeseries'] = cleaned_ts
        item['quality_info'] = quality_info

    quality_df = create_quality_report(file_master_df, loaded_files, quality_params)
    usable_count = quality_df['is_usable'].sum()
    logger.info(f"  ✓ Usable files: {usable_count}/{len(quality_df)}")

    # ------------------------------------------------------------------------
    # 5. Assign Train/Val/Test Splits
    # ------------------------------------------------------------------------
    print("\n[Step 5/8] Assigning train/val/test splits...")

    file_master_df = assign_split_sets(file_master_df, params_config)

    split_counts = file_master_df['split_set'].value_counts()
    logger.info(f"  ✓ Splits: time_split={split_counts.get('time_split', 0)}, "
                f"train={split_counts.get('train', 0)}, "
                f"val={split_counts.get('val', 0)}, "
                f"test={split_counts.get('test', 0)}")

    # ------------------------------------------------------------------------
    # 6. Generate Window Segments
    # ------------------------------------------------------------------------
    print("\n[Step 6/8] Generating window segments...")

    windows_df = create_windows_metadata(loaded_files, file_master_df, params_config)
    logger.info(f"  ✓ Generated {len(windows_df)} windows")

    # Distribution
    for split in ['train', 'val', 'test']:
        split_data = windows_df[windows_df['split_set'] == split]
        n_normal = (split_data['label_binary'] == 1).sum()
        n_abnormal = (split_data['label_binary'] == 0).sum()
        logger.info(f"    {split}: {len(split_data)} windows (Normal: {n_normal}, Abnormal: {n_abnormal})")

    # ------------------------------------------------------------------------
    # 7. Extract 18 Features (9 Basic + 9 Band RMS)
    # ------------------------------------------------------------------------
    print("\n[Step 7/8] Extracting 18 features per window...")

    # Create lookup for timeseries data
    ts_lookup = {item['file_id']: item['timeseries'] for item in loaded_files}

    all_features = []

    for idx, window_row in windows_df.iterrows():
        file_id = window_row['file_id']

        if file_id not in ts_lookup:
            logger.warning(f"Timeseries not found for {file_id}, skipping")
            continue

        ts_df = ts_lookup[file_id]

        # Extract window segment
        idx_start = window_row['idx_start']
        idx_end = window_row['idx_end']
        window_data = ts_df.iloc[idx_start:idx_end + 1]

        if len(window_data) < 100:
            logger.warning(f"Window {window_row['window_id']} too short ({len(window_data)} samples), skipping")
            continue

        # Extract features
        try:
            feats = extract_window_features(window_data, SAMPLING_RATE)

            # Add metadata
            feats['window_id'] = window_row['window_id']
            feats['file_id'] = file_id
            feats['split_set'] = window_row['split_set']
            feats['label_binary'] = window_row['label_binary']
            feats['product'] = window_row['product']
            feats['sample'] = window_row['sample']
            feats['direction'] = window_row['direction']

            all_features.append(feats)

        except Exception as e:
            logger.warning(f"Error extracting features for {window_row['window_id']}: {e}")
            continue

        if (idx + 1) % 100 == 0:
            logger.info(f"    Processed {idx + 1}/{len(windows_df)} windows...")

    # Create DataFrame
    features_df = pd.DataFrame(all_features)
    logger.info(f"  ✓ Extracted features: {len(features_df)} windows × {len(ALL_FEATURES)} features")

    # ------------------------------------------------------------------------
    # 8. Save ECMiner Output CSV
    # ------------------------------------------------------------------------
    print("\n[Step 8/8] Saving ECMiner output CSV...")

    # Remove rows with NaN in feature columns
    clean_df = features_df.dropna(subset=ALL_FEATURES)
    logger.info(f"  ✓ Clean data: {len(clean_df)} windows (removed {len(features_df) - len(clean_df)} with NaN)")

    # Rename split_set → dataset_type (ECMiner convention)
    clean_df = clean_df.rename(columns={'split_set': 'dataset_type'})

    # Reorder columns: features first, then metadata
    metadata_cols = ['window_id', 'file_id', 'dataset_type', 'label_binary', 'product', 'sample', 'direction']
    output_cols = ALL_FEATURES + metadata_cols

    ecminer_df = clean_df[output_cols].copy()

    # Save CSV
    ecminer_df.to_csv(OUTPUT_CSV, index=False)
    logger.info(f"  ✓ Saved to: {OUTPUT_CSV}")

    # ------------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("ECMiner Stage 1 Complete!")
    print("=" * 80)

    print(f"\nOutput: {len(ecminer_df)} windows × {len(ALL_FEATURES)} features")
    print(f"File: {OUTPUT_CSV}")

    print(f"\nDataset Distribution:")
    for dataset in ['train', 'val', 'test']:
        subset = ecminer_df[ecminer_df['dataset_type'] == dataset]
        n_normal = (subset['label_binary'] == 1).sum()
        n_abnormal = (subset['label_binary'] == 0).sum()
        print(f"  {dataset:5s}: {len(subset):3d} windows (Normal: {n_normal:3d}, Abnormal: {n_abnormal:3d})")

    print(f"\nFeatures (18):")
    print(f"  Basic (9):    {', '.join(BASIC_FEATURES)}")
    print(f"  Band RMS (9): {', '.join(BAND_RMS_FEATURES)}")

    print("\n" + "=" * 80)
    print("Next Steps:")
    print("  1. Import ecminer_stage1_output.csv into ECMiner")
    print("  2. Filter: Select 12 features (9 basic + 3 band RMS)")
    print("  3. XGBoost: Train with parameters (n_estimators=100, max_depth=3, etc.)")
    print("  4. Expected Test AUC: ~0.797 (Phase3-1 Clean version)")
    print("=" * 80)

    return ecminer_df


if __name__ == '__main__':
    ecminer_df = main()
