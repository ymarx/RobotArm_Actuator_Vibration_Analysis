"""
Window segmentation for timeseries data
"""
import json
import logging
import random
from typing import Dict, List, Tuple

import pandas as pd

logger = logging.getLogger(__name__)


def get_stable_time_range(
    ts_df: pd.DataFrame,
    margin: float = 0.1
) -> Tuple[float, float]:
    """
    Get stable time range by excluding startup/shutdown periods

    Args:
        ts_df: Timeseries DataFrame with time_sec column
        margin: Fraction to exclude from start and end (e.g., 0.1 = 10%)

    Returns:
        Tuple of (start_time, end_time) for stable region
    """
    if 'time_sec' not in ts_df.columns:
        raise ValueError("time_sec column not found")

    t_min = ts_df['time_sec'].min()
    t_max = ts_df['time_sec'].max()
    span = t_max - t_min

    t_start = t_min + margin * span
    t_end = t_max - margin * span

    return t_start, t_end


def generate_windows_with_constraints(
    ts_df: pd.DataFrame,
    file_info: Dict,
    split_info: Dict,
    params: Dict
) -> List[Dict]:
    """
    Generate windows with constraints (max count, time split awareness)

    Args:
        ts_df: Cleaned timeseries DataFrame with time_sec column
        file_info: File metadata (file_id, product, sample, direction, etc.)
        split_info: Split information (split_set, time_ranges if applicable)
        params: Windowing parameters

    Returns:
        List of window dictionaries
    """
    window_sec = params['windowing']['window_sec']
    hop_sec = params['windowing']['hop_sec']
    margin = params['windowing']['stable_margin']
    max_windows = params['windowing']['max_windows_per_file']
    boundary_handling = params['windowing'].get('boundary_handling', 'strict')

    # Get stable time range
    t_stable_start, t_stable_end = get_stable_time_range(ts_df, margin)

    logger.debug(f"File {file_info['file_id']}: stable range [{t_stable_start:.2f}, {t_stable_end:.2f}]")

    windows = []

    # Check if this is a time-split file (normal samples)
    if split_info['is_time_split']:
        # Generate windows for each split set separately
        time_ranges = split_info['time_ranges']

        for split_name, (frac_start, frac_end) in time_ranges.items():
            # Calculate absolute time range for this split
            total_span = t_stable_end - t_stable_start
            split_start = t_stable_start + frac_start * total_span
            split_end = t_stable_start + frac_end * total_span

            logger.debug(f"  {split_name}: [{split_start:.2f}, {split_end:.2f}]")

            # Generate windows in this range
            cur_t = split_start
            while cur_t + window_sec <= split_end:
                window_end = cur_t + window_sec

                # In strict mode, ensure window doesn't cross split boundary
                if boundary_handling == 'strict' and window_end > split_end:
                    break

                # Get indices for this window
                mask = (ts_df['time_sec'] >= cur_t) & (ts_df['time_sec'] < window_end)
                indices = ts_df.index[mask]

                if len(indices) > 0:
                    windows.append({
                        'file_id': file_info['file_id'],
                        'split_set': split_name,
                        'start_time': cur_t,
                        'end_time': window_end,
                        'idx_start': int(indices[0]),
                        'idx_end': int(indices[-1]),
                        'num_samples': len(indices)
                    })

                cur_t += hop_sec

    else:
        # Regular file-based split (abnormal files)
        cur_t = t_stable_start

        while cur_t + window_sec <= t_stable_end:
            window_end = cur_t + window_sec

            # Get indices for this window
            mask = (ts_df['time_sec'] >= cur_t) & (ts_df['time_sec'] < window_end)
            indices = ts_df.index[mask]

            if len(indices) > 0:
                windows.append({
                    'file_id': file_info['file_id'],
                    'split_set': split_info['split_set'],
                    'start_time': cur_t,
                    'end_time': window_end,
                    'idx_start': int(indices[0]),
                    'idx_end': int(indices[-1]),
                    'num_samples': len(indices)
                })

            cur_t += hop_sec

    # Apply max windows constraint
    if len(windows) > max_windows:
        logger.debug(f"  Limiting windows from {len(windows)} to {max_windows}")
        random.shuffle(windows)
        windows = windows[:max_windows]

    logger.debug(f"  Generated {len(windows)} windows")

    return windows


def create_windows_metadata(
    loaded_files: List[Dict],
    file_master_df: pd.DataFrame,
    params: Dict
) -> pd.DataFrame:
    """
    Create windows metadata table for all files

    Args:
        loaded_files: List of loaded file dictionaries (with cleaned timeseries)
        file_master_df: File master table with split assignments
        params: Configuration parameters

    Returns:
        DataFrame with window metadata
    """
    logger.info("Creating windows metadata")

    all_windows = []

    for item in loaded_files:
        file_id = item['file_id']
        ts_df = item['timeseries']

        # Get file info from master table
        file_row = file_master_df[file_master_df['file_id'] == file_id]

        if len(file_row) == 0:
            logger.warning(f"File {file_id} not found in master table, skipping")
            continue

        file_row = file_row.iloc[0]

        file_info = {
            'file_id': file_id,
            'product': file_row['product'],
            'sample': file_row['sample'],
            'direction': file_row['direction'],
            'label_raw': file_row.get('label_raw'),
            'label_binary': file_row.get('label_binary'),
            'label_weight': file_row.get('label_weight'),
            'is_normal': file_row.get('is_normal')
        }

        # Get split info
        split_info = {
            'split_set': file_row['split_set'],
            'is_time_split': file_row['split_set'] == 'time_split'
        }

        if split_info['is_time_split'] and file_row['time_split_range'] is not None:
            split_info['time_ranges'] = json.loads(file_row['time_split_range'])

        # Generate windows
        windows = generate_windows_with_constraints(ts_df, file_info, split_info, params)

        # Add file metadata to each window
        for win in windows:
            win.update({
                'product': file_info['product'],
                'sample': file_info['sample'],
                'direction': file_info['direction'],
                'label_raw': file_info['label_raw'],
                'label_binary': file_info['label_binary'],
                'label_weight': file_info['label_weight'],
                'is_normal': file_info['is_normal']
            })

        all_windows.extend(windows)

    windows_df = pd.DataFrame(all_windows)

    # Create window_id
    windows_df['window_id'] = [
        f"{row['file_id']}_W{idx:04d}"
        for idx, row in enumerate(all_windows)
    ]

    # Reorder columns
    col_order = [
        'window_id', 'file_id', 'split_set',
        'product', 'sample', 'direction',
        'label_raw', 'label_binary', 'label_weight', 'is_normal',
        'start_time', 'end_time', 'idx_start', 'idx_end', 'num_samples'
    ]

    windows_df = windows_df[[c for c in col_order if c in windows_df.columns]]

    logger.info(f"Created {len(windows_df)} windows")
    logger.info(f"Split distribution:\n{windows_df['split_set'].value_counts()}")

    return windows_df
