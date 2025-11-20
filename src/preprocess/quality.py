"""
Data quality checks
"""
import logging
from typing import Dict, List

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def check_sampling_frequency(
    ts_df: pd.DataFrame,
    expected_fs: float = 512.0,
    tolerance_ratio: float = 0.01
) -> Dict:
    """
    Check if sampling frequency is within expected range

    Args:
        ts_df: Timeseries DataFrame with time_sec column
        expected_fs: Expected sampling frequency in Hz
        tolerance_ratio: Allowed deviation ratio (e.g., 0.01 = ±1%)

    Returns:
        Dictionary with keys:
            - fs_measured: Measured sampling frequency
            - fs_expected: Expected sampling frequency
            - is_valid: Boolean flag
            - delta_ms: Time delta in ms
    """
    if 'time_sec' not in ts_df.columns:
        logger.warning("time_sec column not found, cannot check sampling frequency")
        return {
            'fs_measured': None,
            'fs_expected': expected_fs,
            'is_valid': False,
            'delta_ms': None
        }

    # Calculate time differences
    time_diffs = ts_df['time_sec'].diff().dropna()

    # Get median time delta (in seconds)
    median_dt = time_diffs.median()

    # Calculate sampling frequency
    if median_dt > 0:
        fs_measured = 1.0 / median_dt
    else:
        fs_measured = None

    # Check if within tolerance
    if fs_measured is not None:
        lower_bound = expected_fs * (1 - tolerance_ratio)
        upper_bound = expected_fs * (1 + tolerance_ratio)
        is_valid = lower_bound <= fs_measured <= upper_bound
    else:
        is_valid = False

    return {
        'fs_measured': fs_measured,
        'fs_expected': expected_fs,
        'is_valid': is_valid,
        'delta_ms': median_dt * 1000 if median_dt is not None else None,
        'tolerance_range': [expected_fs * (1 - tolerance_ratio),
                           expected_fs * (1 + tolerance_ratio)]
    }


def check_file_length(
    ts_df: pd.DataFrame,
    expected_range: List[int] = None,
    tolerance_ratio: float = 0.1
) -> Dict:
    """
    Check if file length (number of samples) is within expected range

    Args:
        ts_df: Timeseries DataFrame
        expected_range: [min_samples, max_samples] or None
        tolerance_ratio: Allowed deviation ratio

    Returns:
        Dictionary with keys:
            - num_samples: Actual number of samples
            - expected_range: Expected range
            - is_valid: Boolean flag
    """
    num_samples = len(ts_df)

    if expected_range is None:
        expected_range = [27000, 29000]

    min_expected, max_expected = expected_range

    # Apply tolerance
    min_allowed = min_expected * (1 - tolerance_ratio)
    max_allowed = max_expected * (1 + tolerance_ratio)

    is_valid = min_allowed <= num_samples <= max_allowed

    return {
        'num_samples': num_samples,
        'expected_range': expected_range,
        'allowed_range': [min_allowed, max_allowed],
        'is_valid': is_valid
    }


def check_discontinuities(ts_df: pd.DataFrame) -> Dict:
    """
    Check for discontinuities in timestamp

    Args:
        ts_df: Timeseries DataFrame with time_sec column

    Returns:
        Dictionary with discontinuity information
    """
    if 'time_sec' not in ts_df.columns:
        return {'has_discontinuities': False, 'discontinuity_count': 0}

    time_diffs = ts_df['time_sec'].diff().dropna()
    median_dt = time_diffs.median()

    # Discontinuity: time diff > 2x median
    discontinuities = time_diffs > (2 * median_dt)

    return {
        'has_discontinuities': discontinuities.any(),
        'discontinuity_count': discontinuities.sum(),
        'discontinuity_ratio': discontinuities.mean()
    }


def run_quality_checks(
    ts_df: pd.DataFrame,
    params: Dict = None
) -> Dict:
    """
    Run complete quality checks on timeseries

    Args:
        ts_df: Timeseries DataFrame
        params: Quality check parameters

    Returns:
        Dictionary with all quality check results
    """
    if params is None:
        params = {
            'expected_fs': 512.0,
            'fs_tolerance': 0.01,
            'expected_length': [27000, 29000],
            'length_tolerance': 0.1
        }

    results = {}

    # Check sampling frequency
    results['sampling_frequency'] = check_sampling_frequency(
        ts_df,
        expected_fs=params.get('expected_fs', 512.0),
        tolerance_ratio=params.get('fs_tolerance', 0.01)
    )

    # Check file length
    results['file_length'] = check_file_length(
        ts_df,
        expected_range=params.get('expected_length', [27000, 29000]),
        tolerance_ratio=params.get('length_tolerance', 0.1)
    )

    # Check discontinuities
    results['discontinuities'] = check_discontinuities(ts_df)

    # Overall quality flag
    results['is_usable'] = (
        results['sampling_frequency']['is_valid'] and
        results['file_length']['is_valid'] and
        not results['discontinuities']['has_discontinuities']
    )

    return results


def create_quality_report(
    file_master_df: pd.DataFrame,
    loaded_files: List[Dict],
    params: Dict = None
) -> pd.DataFrame:
    """
    Create quality report for all files

    Args:
        file_master_df: File master table
        loaded_files: List of loaded file dictionaries
        params: Quality check parameters

    Returns:
        DataFrame with quality check results for each file
    """
    logger.info("Creating quality report for all files")

    quality_records = []

    for item in loaded_files:
        file_id = item['file_id']
        ts_df = item['timeseries']

        # Get product from file_id (100W or 200W)
        product = file_id.split('_')[0]

        # Adjust expected_length based on product
        qc_params = {}
        if params:
            # Extract and restructure params for run_quality_checks
            qc_params['expected_fs'] = params.get('sampling_freq', {}).get('expected', 512.0)
            qc_params['fs_tolerance'] = params.get('sampling_freq', {}).get('tolerance_ratio', 0.01)
            qc_params['length_tolerance'] = params.get('file_length', {}).get('tolerance_ratio', 0.1)

            # Get product-specific expected length
            expected_samples = params.get('file_length', {}).get('expected_samples', [27000, 29000])
            if isinstance(expected_samples, dict):
                # Product-specific configuration
                qc_params['expected_length'] = expected_samples.get(product, [27000, 29000])
            else:
                # Legacy single-value configuration
                qc_params['expected_length'] = expected_samples

        # Run quality checks
        qc_results = run_quality_checks(ts_df, qc_params)

        record = {
            'file_id': file_id,
            'num_samples': len(ts_df),
            'fs_measured': qc_results['sampling_frequency']['fs_measured'],
            'fs_is_valid': qc_results['sampling_frequency']['is_valid'],
            'length_is_valid': qc_results['file_length']['is_valid'],
            'has_discontinuities': qc_results['discontinuities']['has_discontinuities'],
            'discontinuity_count': qc_results['discontinuities']['discontinuity_count'],
            'is_usable': qc_results['is_usable']
        }

        quality_records.append(record)

    quality_df = pd.DataFrame(quality_records)

    # Summary
    logger.info(f"Quality report summary:")
    logger.info(f"  Total files: {len(quality_df)}")
    logger.info(f"  Usable files: {quality_df['is_usable'].sum()}")
    logger.info(f"  Files with issues: {(~quality_df['is_usable']).sum()}")

    return quality_df
