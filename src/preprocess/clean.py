"""
Data cleaning and preprocessing
"""
import logging
from typing import Tuple, Dict

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Sensor columns
SENSOR_COLS = ['acc-X', 'acc-Y', 'acc-Z', 'Gyro-X', 'Gyro-Y', 'Gyro-Z', 'acc-Sum']


def normalize_timestamp(ts_df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert TimeStamp to relative time in seconds

    Args:
        ts_df: Timeseries DataFrame with TimeStamp column

    Returns:
        DataFrame with added 'time_sec' column
    """
    ts_df = ts_df.copy()

    # TimeStamp is in milliseconds, convert to seconds relative to first sample
    t0 = ts_df['TimeStamp'].iloc[0]
    ts_df['time_sec'] = (ts_df['TimeStamp'] - t0) / 1000.0

    return ts_df


def handle_missing_values(
    ts_df: pd.DataFrame,
    max_nan_ratio: float = 0.05,
    interpolation_method: str = 'linear'
) -> Tuple[pd.DataFrame, Dict]:
    """
    Handle missing values in sensor channels

    Args:
        ts_df: Timeseries DataFrame
        max_nan_ratio: Maximum allowed NaN ratio per channel (0-1)
        interpolation_method: Method for interpolation ('linear', 'cubic', etc.)

    Returns:
        Tuple of (cleaned DataFrame, quality info dict)
    """
    ts_df = ts_df.copy()

    # Check which sensor columns exist
    available_sensors = [col for col in SENSOR_COLS if col in ts_df.columns]

    if not available_sensors:
        logger.warning("No sensor columns found")
        return ts_df, {}

    # Calculate NaN ratio per channel
    nan_ratio = ts_df[available_sensors].isna().mean()

    bad_channels = nan_ratio[nan_ratio > max_nan_ratio].index.tolist()

    quality_info = {
        'nan_ratio': nan_ratio.to_dict(),
        'bad_channels': bad_channels,
        'has_quality_issues': len(bad_channels) > 0
    }

    if bad_channels:
        logger.warning(f"Channels exceeding NaN threshold: {bad_channels}")

    # Interpolate NaN values
    ts_df[available_sensors] = ts_df[available_sensors].interpolate(
        method=interpolation_method,
        limit_direction='both'
    )

    # Fill any remaining NaN with forward/backward fill
    ts_df[available_sensors] = ts_df[available_sensors].ffill().bfill()

    return ts_df, quality_info


def detect_and_fix_spikes(
    ts_df: pd.DataFrame,
    z_threshold: float = 8.0
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Detect and fix spikes/outliers in sensor data

    Args:
        ts_df: Timeseries DataFrame
        z_threshold: Z-score threshold for spike detection

    Returns:
        Tuple of (cleaned DataFrame, spike mask DataFrame)
    """
    ts_df = ts_df.copy()

    # Check which sensor columns exist
    available_sensors = [col for col in SENSOR_COLS if col in ts_df.columns]

    if not available_sensors:
        logger.warning("No sensor columns found")
        return ts_df, pd.DataFrame()

    spike_mask = pd.DataFrame(False, index=ts_df.index, columns=available_sensors)

    for col in available_sensors:
        x = ts_df[col].values

        # Calculate z-score
        mean = np.mean(x)
        std = np.std(x)

        if std == 0:
            logger.warning(f"Column {col} has zero std, skipping spike detection")
            continue

        z_scores = np.abs((x - mean) / std)

        # Detect spikes
        spikes = z_scores > z_threshold
        spike_mask[col] = spikes

        # Mark spikes as NaN
        ts_df.loc[spikes, col] = np.nan

    # Interpolate spike values
    ts_df[available_sensors] = ts_df[available_sensors].interpolate(
        method='linear',
        limit_direction='both'
    ).ffill().bfill()

    return ts_df, spike_mask


def clean_timeseries(
    ts_df: pd.DataFrame,
    params: Dict = None
) -> Tuple[pd.DataFrame, Dict]:
    """
    Complete timeseries cleaning pipeline

    Args:
        ts_df: Raw timeseries DataFrame
        params: Parameters dictionary with keys:
            - max_nan_ratio: float
            - interpolation_method: str
            - z_threshold: float

    Returns:
        Tuple of (cleaned DataFrame, quality info dict)
    """
    if params is None:
        params = {
            'max_nan_ratio': 0.05,
            'interpolation_method': 'linear',
            'z_threshold': 8.0
        }

    quality_info = {}

    # Step 1: Normalize timestamp
    ts_df = normalize_timestamp(ts_df)

    # Step 2: Handle missing values
    ts_df, nan_info = handle_missing_values(
        ts_df,
        max_nan_ratio=params.get('max_nan_ratio', 0.05),
        interpolation_method=params.get('interpolation_method', 'linear')
    )
    quality_info['missing_values'] = nan_info

    # Step 3: Detect and fix spikes
    ts_df, spike_mask = detect_and_fix_spikes(
        ts_df,
        z_threshold=params.get('z_threshold', 8.0)
    )

    # Calculate spike ratio per channel
    available_sensors = [col for col in SENSOR_COLS if col in ts_df.columns]
    spike_ratio = spike_mask[available_sensors].mean()

    quality_info['spikes'] = {
        'spike_ratio': spike_ratio.to_dict(),
        'total_spikes': spike_mask[available_sensors].sum().to_dict()
    }

    return ts_df, quality_info
