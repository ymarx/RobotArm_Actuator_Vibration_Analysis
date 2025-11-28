"""
Class balancing for training data
"""
import logging
from typing import Dict, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def calculate_class_distribution(windows_df: pd.DataFrame) -> Dict:
    """
    Calculate class distribution across split sets

    Args:
        windows_df: Windows metadata DataFrame

    Returns:
        Dictionary with distribution statistics
    """
    distribution = {}

    for split_set in ['train', 'val', 'test']:
        split_data = windows_df[windows_df['split_set'] == split_set]

        if len(split_data) == 0:
            distribution[split_set] = {
                'total': 0,
                'normal': 0,
                'abnormal': 0,
                'ratio': None
            }
            continue

        normal_count = (split_data['label_binary'] == 1).sum()
        abnormal_count = (split_data['label_binary'] == 0).sum()

        distribution[split_set] = {
            'total': len(split_data),
            'normal': normal_count,
            'abnormal': abnormal_count,
            'ratio': normal_count / abnormal_count if abnormal_count > 0 else None
        }

    return distribution


def balance_train_windows(
    windows_df: pd.DataFrame,
    params: Dict
) -> pd.DataFrame:
    """
    Balance train set windows to achieve target class ratio

    Args:
        windows_df: Windows metadata DataFrame
        params: Balancing parameters

    Returns:
        Balanced DataFrame (only train set modified, val/test unchanged)
    """
    logger.info("Balancing train windows")

    # Get balancing config
    balance_config = params.get('balancing', {})
    target_ratio = balance_config.get('target_ratio', {'normal': 1, 'abnormal': 2})
    method = balance_config.get('method', 'none')
    random_seed = balance_config.get('random_seed', 42)

    # If balancing is disabled, return original data
    if method == 'none':
        train_df = windows_df[windows_df['split_set'] == 'train']
        n_normal = (train_df['label_binary'] == 1).sum()
        n_abnormal = (train_df['label_binary'] == 0).sum()
        logger.info(f"Balancing disabled (method='none')")
        logger.info(f"Original train distribution preserved: normal={n_normal}, abnormal={n_abnormal}")
        return windows_df

    np.random.seed(random_seed)

    # Split data
    train_df = windows_df[windows_df['split_set'] == 'train'].copy()
    other_df = windows_df[windows_df['split_set'] != 'train'].copy()

    if len(train_df) == 0:
        logger.warning("No train windows found, skipping balancing")
        return windows_df

    # Get current distribution
    normal_windows = train_df[train_df['label_binary'] == 1]
    abnormal_windows = train_df[train_df['label_binary'] == 0]

    n_normal = len(normal_windows)
    n_abnormal = len(abnormal_windows)

    logger.info(f"Current train distribution: normal={n_normal}, abnormal={n_abnormal}")

    if n_normal == 0 or n_abnormal == 0:
        logger.warning("One class has no samples, skipping balancing")
        return windows_df

    # Calculate target counts
    ratio_normal = target_ratio.get('normal', 1)
    ratio_abnormal = target_ratio.get('abnormal', 2)

    # Target: ratio_normal : ratio_abnormal
    # Calculate target counts based on method
    if method == 'oversample':
        # Increase minority class
        if n_normal * ratio_abnormal < n_abnormal * ratio_normal:
            # Normal is minority
            n_normal_target = int(n_abnormal * ratio_normal / ratio_abnormal)
            n_abnormal_target = n_abnormal
        else:
            # Abnormal is minority
            n_normal_target = n_normal
            n_abnormal_target = int(n_normal * ratio_abnormal / ratio_normal)

    elif method == 'undersample':
        # Decrease majority class
        if n_normal * ratio_abnormal < n_abnormal * ratio_normal:
            # Normal is minority
            n_normal_target = n_normal
            n_abnormal_target = int(n_normal * ratio_abnormal / ratio_normal)
        else:
            # Abnormal is minority
            n_normal_target = int(n_abnormal * ratio_normal / ratio_abnormal)
            n_abnormal_target = n_abnormal

    else:
        logger.warning(f"Unknown balancing method: {method}, skipping")
        return windows_df

    logger.info(f"Target distribution: normal={n_normal_target}, abnormal={n_abnormal_target}")

    # Resample normal class
    if n_normal_target > n_normal:
        # Oversample
        normal_resampled = normal_windows.sample(
            n=n_normal_target,
            replace=True,
            random_state=random_seed
        )
    elif n_normal_target < n_normal:
        # Undersample
        normal_resampled = normal_windows.sample(
            n=n_normal_target,
            replace=False,
            random_state=random_seed
        )
    else:
        normal_resampled = normal_windows

    # Resample abnormal class
    if n_abnormal_target > n_abnormal:
        # Oversample
        abnormal_resampled = abnormal_windows.sample(
            n=n_abnormal_target,
            replace=True,
            random_state=random_seed
        )
    elif n_abnormal_target < n_abnormal:
        # Undersample
        abnormal_resampled = abnormal_windows.sample(
            n=n_abnormal_target,
            replace=False,
            random_state=random_seed
        )
    else:
        abnormal_resampled = abnormal_windows

    # Combine
    balanced_train = pd.concat([normal_resampled, abnormal_resampled], ignore_index=True)

    # Shuffle
    balanced_train = balanced_train.sample(frac=1, random_state=random_seed).reset_index(drop=True)

    logger.info(f"Balanced train: normal={len(normal_resampled)}, abnormal={len(abnormal_resampled)}")

    # Combine with val/test
    result = pd.concat([balanced_train, other_df], ignore_index=True)

    # ========================================================================
    # CRITICAL VALIDATION: Check for duplicate window_ids
    # ========================================================================
    if 'window_id' in result.columns:
        n_total = len(result)
        n_unique = result['window_id'].nunique()
        n_duplicates = n_total - n_unique

        if n_duplicates > 0:
            logger.error(f"CRITICAL BUG: {n_duplicates} duplicate window_ids detected after balancing!")

            # Log details of duplicates
            duplicated_ids = result[result['window_id'].duplicated(keep=False)]['window_id'].unique()
            logger.error(f"Duplicate window_ids (first 10): {duplicated_ids[:10].tolist()}")

            # Count duplicates by split_set
            for split in ['train', 'val', 'test']:
                split_data = result[result['split_set'] == split]
                split_dups = split_data['window_id'].duplicated().sum()
                if split_dups > 0:
                    logger.error(f"  {split}: {split_dups} duplicates")

            raise ValueError(
                f"Data integrity violation: {n_duplicates} duplicate window_ids found. "
                "This typically indicates oversampling with replace=True. "
                "Check balancing configuration and ensure method='none' or unique ID generation."
            )

        logger.info(f"✅ Validation passed: {n_unique} unique window_ids (no duplicates)")

    return result


def get_balancing_report(
    windows_df: pd.DataFrame,
    balanced_df: pd.DataFrame
) -> Dict:
    """
    Create report comparing original and balanced distributions

    Args:
        windows_df: Original windows DataFrame
        balanced_df: Balanced windows DataFrame

    Returns:
        Dictionary with before/after statistics
    """
    report = {
        'before': calculate_class_distribution(windows_df),
        'after': calculate_class_distribution(balanced_df)
    }

    return report
