"""
Data splitting strategies for train/val/test
"""
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def create_time_based_splits(
    file_master_df: pd.DataFrame,
    params: Dict
) -> pd.DataFrame:
    """
    Create time-based splits for normal samples (limited normal data strategy)

    For normal samples: Split file internally by time ranges
    For abnormal samples: Split by files

    Args:
        file_master_df: File master table with is_normal flag
        params: Split parameters from config

    Returns:
        DataFrame with 'split_set' and 'time_split_range' columns added
    """
    logger.info("Creating time-based splits")

    file_master_df = file_master_df.copy()

    # Get time ranges from config
    time_config = params['split']['time_based']
    train_range = time_config['train_range']
    val_range = time_config['val_range']
    test_range = time_config['test_range']

    # Initialize columns
    file_master_df['split_set'] = None
    file_master_df['time_split_range'] = None

    # Process normal files (time-based split)
    normal_mask = file_master_df['is_normal'] == True

    logger.info(f"Normal files: {normal_mask.sum()}")

    for idx, row in file_master_df[normal_mask].iterrows():
        # Each normal file will be split into train/val/test by time ranges
        # We'll mark it with a special flag and handle actual splitting in segment generation
        file_master_df.at[idx, 'split_set'] = 'time_split'
        file_master_df.at[idx, 'time_split_range'] = json.dumps({
            'train': train_range,
            'val': val_range,
            'test': test_range
        })

    # Process abnormal files (file-based split)
    abnormal_mask = file_master_df['is_normal'] == False

    logger.info(f"Abnormal files: {abnormal_mask.sum()}")

    if abnormal_mask.sum() > 0:
        abnormal_files = file_master_df[abnormal_mask].copy()

        # Split by product to maintain balance
        for product in abnormal_files['product'].unique():
            product_mask = abnormal_files['product'] == product

            product_files = abnormal_files[product_mask].copy()
            n_files = len(product_files)

            logger.info(f"  {product}: {n_files} abnormal files")

            if n_files == 0:
                continue

            # Shuffle and split
            random_seed = time_config['abnormal_split'].get('random_seed', 42)
            np.random.seed(random_seed)

            indices = np.random.permutation(n_files)

            train_ratio = time_config['abnormal_split']['train_ratio']
            val_ratio = time_config['abnormal_split']['val_ratio']

            n_train = int(n_files * train_ratio)
            n_val = int(n_files * val_ratio)

            train_idx = indices[:n_train]
            val_idx = indices[n_train:n_train + n_val]
            test_idx = indices[n_train + n_val:]

            # Assign splits
            file_ids = product_files.index.tolist()

            for i, file_idx in enumerate(file_ids):
                if i in train_idx:
                    file_master_df.at[file_idx, 'split_set'] = 'train'
                elif i in val_idx:
                    file_master_df.at[file_idx, 'split_set'] = 'val'
                elif i in test_idx:
                    file_master_df.at[file_idx, 'split_set'] = 'test'

    # Summary
    logger.info("Split summary:")
    logger.info(f"  time_split (normal): {(file_master_df['split_set'] == 'time_split').sum()}")
    logger.info(f"  train (abnormal): {(file_master_df['split_set'] == 'train').sum()}")
    logger.info(f"  val (abnormal): {(file_master_df['split_set'] == 'val').sum()}")
    logger.info(f"  test (abnormal): {(file_master_df['split_set'] == 'test').sum()}")

    return file_master_df


def create_file_based_splits(
    file_master_df: pd.DataFrame,
    params: Dict
) -> pd.DataFrame:
    """
    Create file-based splits (for when sufficient normal files exist)

    Args:
        file_master_df: File master table
        params: Split parameters

    Returns:
        DataFrame with 'split_set' column added
    """
    logger.info("Creating file-based splits")

    # This would be used if we had more normal samples
    # For now, not implemented since we only have 1 normal sample per product
    logger.warning("File-based split not fully implemented - use time-based split instead")

    file_master_df = file_master_df.copy()
    file_master_df['split_set'] = 'train'  # Default all to train

    return file_master_df


def assign_split_sets(
    file_master_df: pd.DataFrame,
    params: Dict,
    output_path: Path = None
) -> pd.DataFrame:
    """
    Assign train/val/test split sets to files based on strategy

    Args:
        file_master_df: File master table
        params: Configuration parameters
        output_path: Path to save split mapping (optional)

    Returns:
        DataFrame with split assignments
    """
    strategy = params['split']['strategy']

    logger.info(f"Using split strategy: {strategy}")

    if strategy == 'time_based':
        file_master_df = create_time_based_splits(file_master_df, params)
    elif strategy == 'file_based':
        file_master_df = create_file_based_splits(file_master_df, params)
    else:
        raise ValueError(f"Unknown split strategy: {strategy}")

    # Save split mapping
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)

        split_mapping = file_master_df[['file_id', 'split_set', 'time_split_range']].to_dict('records')

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump({
                'strategy': strategy,
                'mapping': split_mapping
            }, f, indent=2, ensure_ascii=False)

        logger.info(f"Split mapping saved to {output_path}")

    return file_master_df


def get_file_split_info(file_id: str, file_master_df: pd.DataFrame) -> Dict:
    """
    Get split information for a specific file

    Args:
        file_id: File identifier
        file_master_df: File master table with split assignments

    Returns:
        Dictionary with split information
    """
    file_row = file_master_df[file_master_df['file_id'] == file_id]

    if len(file_row) == 0:
        return {'split_set': None, 'is_time_split': False}

    row = file_row.iloc[0]

    result = {
        'split_set': row['split_set'],
        'is_time_split': row['split_set'] == 'time_split'
    }

    if result['is_time_split'] and row['time_split_range'] is not None:
        result['time_ranges'] = json.loads(row['time_split_range'])

    return result
