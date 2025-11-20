"""
Load labels from Excel file and create file master table
"""
import logging
from pathlib import Path
from typing import Dict, Optional

import pandas as pd

from ..utils.helpers import load_config


logger = logging.getLogger(__name__)


def load_labels_from_excel(excel_path: Path) -> pd.DataFrame:
    """
    Load labels from Excel file

    Expected structure:
        - Product column: 100W or 200W
        - Sample column: Sample number (e.g., 0, 1, 2, ...)
        - Label column: 정상, 소음, 진동, 표기없음

    Args:
        excel_path: Path to Excel file

    Returns:
        DataFrame with columns: product, sample, label
    """
    logger.info(f"Loading labels from {excel_path}")

    try:
        # Read Excel file without header
        df = pd.read_excel(excel_path, sheet_name=0, header=None)

        logger.debug(f"Excel shape: {df.shape}")

        # Find header row (contains "시료번호" or "100W")
        header_row_idx = None
        for idx, row in df.iterrows():
            row_str = ' '.join(row.astype(str).tolist())
            if '시료번호' in row_str or ('100W' in row_str and '200W' in row_str):
                header_row_idx = idx
                break

        if header_row_idx is None:
            raise ValueError("Could not find header row")

        logger.debug(f"Found header at row {header_row_idx}")

        # Get data rows
        df_data = df.iloc[header_row_idx + 1:].reset_index(drop=True)
        df_data.columns = df.iloc[header_row_idx].tolist()

        # Find columns
        sample_col = None
        col_100w = None
        col_200w = None

        for col in df_data.columns:
            col_str = str(col).strip()
            if '시료' in col_str or '번호' in col_str:
                sample_col = col
            elif '100' in col_str:
                col_100w = col
            elif '200' in col_str:
                col_200w = col

        logger.info(f"Columns: sample={sample_col}, 100W={col_100w}, 200W={col_200w}")

        # Convert to long format
        records = []
        for idx, row in df_data.iterrows():
            sample_num = row[sample_col]
            if pd.isna(sample_num):
                continue
            try:
                sample_num = int(float(sample_num))
            except:
                continue

            # 100W label
            if col_100w is not None and not pd.isna(row[col_100w]):
                label = str(row[col_100w]).strip()
                if label and label != 'nan':
                    records.append({'product': '100W', 'sample': sample_num, 'label': label})

            # 200W label
            if col_200w is not None and not pd.isna(row[col_200w]):
                label = str(row[col_200w]).strip()
                if label and label != 'nan':
                    records.append({'product': '200W', 'sample': sample_num, 'label': label})

        labels_df = pd.DataFrame(records)

        # Normalize labels
        valid_labels = ['정상', '소음', '진동', '표기없음']
        labels_df['label'] = labels_df['label'].apply(lambda x: x if x in valid_labels else None)
        labels_df = labels_df.dropna(subset=['label'])

        logger.info(f"Loaded {len(labels_df)} label records")
        logger.info(f"Distribution:\n{labels_df.groupby(['product', 'label']).size()}")

        return labels_df

    except Exception as e:
        logger.error(f"Error loading labels from Excel: {str(e)}")
        raise


def create_file_master_table(
    metadata_df: pd.DataFrame,
    labels_df: pd.DataFrame,
    params_config: Optional[Dict] = None
) -> pd.DataFrame:
    """
    Create file master table by merging metadata and labels

    Args:
        metadata_df: DataFrame from create_metadata_table
        labels_df: DataFrame from load_labels_from_excel
        params_config: Parameters config (for special label overrides)

    Returns:
        Merged DataFrame with additional columns:
            - label_raw: Original label
            - label_binary: PASS(1) / FAIL(0)
            - label_weight: Sample weight for training
            - is_normal: Boolean flag for normal samples
    """
    logger.info("Creating file master table")

    # Load params if not provided
    if params_config is None:
        params_config = load_config('params_eda')

    # Merge metadata with labels on (product, sample)
    file_master = metadata_df.merge(
        labels_df,
        on=['product', 'sample'],
        how='left'
    )

    logger.info(f"Merged {len(file_master)} files")

    # Check for unmatched files
    unmatched = file_master['label'].isna().sum()
    if unmatched > 0:
        logger.warning(f"{unmatched} files have no label match")
        logger.debug(f"Unmatched files:\n{file_master[file_master['label'].isna()][['file_id', 'product', 'sample']]}")

    # Apply special label overrides from config
    if 'special_override' in params_config.get('labels', {}):
        for override in params_config['labels']['special_override']:
            mask = (
                (file_master['product'] == override['product']) &
                (file_master['sample'] == override['sample'])
            )
            if mask.any():
                logger.info(f"Applying override: {override['product']} Sample{override['sample']}: "
                          f"{override['original_label']} → {override['override_label']}")
                file_master.loc[mask, 'label'] = override['override_label']

    # Rename label to label_raw
    file_master.rename(columns={'label': 'label_raw'}, inplace=True)

    # Create binary label (PASS=1, FAIL=0)
    label_map = params_config.get('labels', {}).get('binary_map', {})
    pass_labels = label_map.get('PASS', ['정상', '소음'])
    fail_labels = label_map.get('FAIL', ['진동', '표기없음'])

    file_master['label_binary'] = file_master['label_raw'].apply(
        lambda x: 1 if x in pass_labels else (0 if x in fail_labels else None)
    )

    # Create label weight (소음은 weak positive)
    weak_weight = params_config.get('labels', {}).get('weak_positive_weight', 0.5)

    def get_weight(row):
        if row['label_raw'] == '소음':
            # Check if it's a special override (200W Sample03)
            if row['product'] == '200W' and row['sample'] == 3:
                return 1.0  # Full weight for overridden normal
            else:
                return weak_weight  # Weak positive weight
        else:
            return 1.0

    file_master['label_weight'] = file_master.apply(get_weight, axis=1)

    # Create is_normal flag (for train/val/test split logic)
    normal_samples = params_config.get('normal_samples', {})
    file_master['is_normal'] = file_master.apply(
        lambda row: row['sample'] in normal_samples.get(row['product'], []),
        axis=1
    )

    # Add usability flag (default True, will be updated by quality checks)
    file_master['is_usable'] = True
    file_master['qc_notes'] = ''

    logger.info(f"File master table created with {len(file_master)} records")
    logger.info(f"Label distribution:\n{file_master['label_raw'].value_counts()}")
    logger.info(f"Binary label distribution:\n{file_master['label_binary'].value_counts()}")
    logger.info(f"Normal samples: {file_master['is_normal'].sum()}")

    return file_master
