"""
Time domain feature extraction
"""
import logging
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew

logger = logging.getLogger(__name__)

# Sensor columns
SENSOR_COLS = ['acc-X', 'acc-Y', 'acc-Z', 'Gyro-X', 'Gyro-Y', 'Gyro-Z', 'acc-Sum']


def compute_time_domain_features_for_window(
    ts_df: pd.DataFrame,
    window_meta: Dict,
    params: Dict = None
) -> Dict:
    """
    Compute time domain features for a single window

    Args:
        ts_df: Complete timeseries DataFrame
        window_meta: Window metadata with idx_start, idx_end
        params: Feature extraction parameters

    Returns:
        Dictionary of features
    """
    # Extract window data
    idx_start = window_meta['idx_start']
    idx_end = window_meta['idx_end']

    window_data = ts_df.iloc[idx_start:idx_end + 1].copy()

    features = {}

    # Add metadata
    features['window_id'] = window_meta['window_id']
    features['file_id'] = window_meta['file_id']
    features['split_set'] = window_meta['split_set']

    # Product and direction
    features['product'] = window_meta['product']
    features['sample'] = window_meta['sample']
    features['direction'] = window_meta['direction']

    # Direction binary feature
    features['direction_cw'] = 1 if window_meta['direction'] == 'CW' else 0

    # Product binary feature (for combined model)
    features['product_100w'] = 1 if window_meta['product'] == '100W' else 0

    # Labels
    features['label_raw'] = window_meta.get('label_raw')
    features['label_binary'] = window_meta.get('label_binary')
    features['label_weight'] = window_meta.get('label_weight')

    # Check which sensor columns exist
    available_sensors = [col for col in SENSOR_COLS if col in window_data.columns]

    if not available_sensors:
        logger.warning(f"No sensor columns found for window {window_meta['window_id']}")
        return features

    # Compute features for each channel
    for col in available_sensors:
        x = window_data[col].values

        if len(x) == 0:
            continue

        col_prefix = col.replace('-', '_')

        # Basic statistics
        features[f'{col_prefix}_mean'] = float(np.mean(x))
        features[f'{col_prefix}_std'] = float(np.std(x))

        # RMS (Root Mean Square)
        features[f'{col_prefix}_rms'] = float(np.sqrt(np.mean(x ** 2)))

        # Peak (maximum absolute value)
        features[f'{col_prefix}_peak'] = float(np.max(np.abs(x)))

        # Crest Factor = Peak / RMS
        rms_val = features[f'{col_prefix}_rms']
        if rms_val > 0:
            features[f'{col_prefix}_crest'] = features[f'{col_prefix}_peak'] / rms_val
        else:
            features[f'{col_prefix}_crest'] = 0.0

        # Kurtosis (tailedness of distribution)
        if len(x) > 3:  # Need at least 4 samples for kurtosis
            features[f'{col_prefix}_kurtosis'] = float(kurtosis(x, fisher=True, bias=False, nan_policy='omit'))
        else:
            features[f'{col_prefix}_kurtosis'] = 0.0

        # Skewness (asymmetry of distribution)
        if len(x) > 2:  # Need at least 3 samples for skewness
            features[f'{col_prefix}_skewness'] = float(skew(x, bias=False, nan_policy='omit'))
        else:
            features[f'{col_prefix}_skewness'] = 0.0

    # Window-level metadata
    features['window_duration'] = window_meta['end_time'] - window_meta['start_time']
    features['window_num_samples'] = window_meta['num_samples']

    return features


def extract_features_from_windows(
    loaded_files: List[Dict],
    windows_df: pd.DataFrame,
    params: Dict = None
) -> pd.DataFrame:
    """
    Extract features from all windows

    Args:
        loaded_files: List of loaded file dictionaries (with cleaned timeseries)
        windows_df: Windows metadata DataFrame
        params: Feature extraction parameters

    Returns:
        DataFrame of features
    """
    logger.info(f"Extracting features from {len(windows_df)} windows")

    # Create file_id -> timeseries mapping
    ts_lookup = {item['file_id']: item['timeseries'] for item in loaded_files}

    all_features = []

    for idx, window_row in windows_df.iterrows():
        file_id = window_row['file_id']

        if file_id not in ts_lookup:
            logger.warning(f"Timeseries not found for file {file_id}, skipping window")
            continue

        ts_df = ts_lookup[file_id]

        # Extract features
        try:
            features = compute_time_domain_features_for_window(
                ts_df,
                window_row.to_dict(),
                params
            )
            all_features.append(features)

        except Exception as e:
            logger.error(f"Error extracting features for window {window_row['window_id']}: {str(e)}")
            continue

        if (idx + 1) % 100 == 0:
            logger.info(f"  Processed {idx + 1}/{len(windows_df)} windows")

    features_df = pd.DataFrame(all_features)

    logger.info(f"Extracted features: {len(features_df)} windows × {len(features_df.columns)} features")

    return features_df
