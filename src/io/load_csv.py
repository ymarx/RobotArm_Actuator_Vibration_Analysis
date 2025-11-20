"""
CSV file loading and metadata extraction
"""
import io
import logging
from pathlib import Path
from typing import Dict, Tuple, List

import pandas as pd

from ..utils.helpers import parse_filename, create_file_id


logger = logging.getLogger(__name__)


def parse_csv_with_metadata(csv_path: Path) -> Tuple[Dict, pd.DataFrame]:
    """
    Parse CSV file to extract metadata and timeseries data

    CSV Structure:
        - Top section: Metadata (key,value pairs)
        - "DataSet" marker line
        - Header line: TimeStamp,acc-X,acc-Y,acc-Z,Gyro-X,Gyro-Y,Gyro-Z,acc-Sum
        - Data rows

    Args:
        csv_path: Path to CSV file

    Returns:
        Tuple of (metadata_dict, timeseries_dataframe)
    """
    logger.debug(f"Parsing CSV: {csv_path.name}")

    try:
        # Read entire file
        with open(csv_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        metadata = {}
        data_lines = []
        in_dataset = False

        for line in lines:
            line = line.strip()

            if not line:
                continue

            # Check for DataSet marker
            if line.lower().startswith('dataset'):
                in_dataset = True
                continue

            if not in_dataset:
                # Parse metadata: "Key,Value"
                if ',' in line:
                    parts = line.split(',', 1)
                    if len(parts) == 2:
                        key, value = parts
                        metadata[key.strip()] = value.strip()
            else:
                # Collect data lines (including header)
                data_lines.append(line)

        if not data_lines:
            raise ValueError(f"No data found after DataSet marker in {csv_path.name}")

        # Parse timeseries data
        ts_df = pd.read_csv(io.StringIO('\n'.join(data_lines)))

        # Validate expected columns
        expected_cols = ['TimeStamp', 'acc-X', 'acc-Y', 'acc-Z',
                        'Gyro-X', 'Gyro-Y', 'Gyro-Z', 'acc-Sum']
        missing_cols = set(expected_cols) - set(ts_df.columns)
        if missing_cols:
            logger.warning(f"Missing columns in {csv_path.name}: {missing_cols}")

        logger.debug(f"Parsed {len(ts_df)} samples with {len(metadata)} metadata fields")

        return metadata, ts_df

    except Exception as e:
        logger.error(f"Error parsing {csv_path.name}: {str(e)}")
        raise


def load_all_csv_files(csv_dir: Path, product: str) -> List[Dict]:
    """
    Load all CSV files from directory

    Args:
        csv_dir: Directory containing CSV files
        product: Product type ('100W' or '200W')

    Returns:
        List of dictionaries, each containing:
            - file_id: Unique identifier
            - file_path: Path to CSV
            - filename: Filename
            - metadata: Dictionary of metadata
            - timeseries: DataFrame of timeseries
            - parsed_filename: Parsed filename info
    """
    logger.info(f"Loading CSV files from {csv_dir} for product {product}")

    csv_files = sorted(csv_dir.glob("*.csv"))
    logger.info(f"Found {len(csv_files)} CSV files")

    results = []

    for csv_path in csv_files:
        try:
            # Parse filename
            parsed = parse_filename(csv_path.name)

            # Skip if product doesn't match
            if parsed['product'] != product:
                logger.debug(f"Skipping {csv_path.name} (product mismatch)")
                continue

            # Parse CSV
            metadata, timeseries = parse_csv_with_metadata(csv_path)

            # Create file ID
            if all(parsed[k] is not None for k in ['product', 'sample', 'direction', 'run_id']):
                file_id = create_file_id(
                    parsed['product'],
                    parsed['sample'],
                    parsed['direction'],
                    parsed['run_id']
                )
            else:
                file_id = csv_path.stem  # Fallback to filename without extension

            results.append({
                'file_id': file_id,
                'file_path': str(csv_path),
                'filename': csv_path.name,
                'metadata': metadata,
                'timeseries': timeseries,
                'parsed_filename': parsed
            })

            logger.debug(f"Loaded: {file_id} ({len(timeseries)} samples)")

        except Exception as e:
            logger.error(f"Failed to load {csv_path.name}: {str(e)}")
            continue

    logger.info(f"Successfully loaded {len(results)} files for {product}")

    return results


def create_metadata_table(loaded_files: List[Dict]) -> pd.DataFrame:
    """
    Create metadata table from loaded files

    Args:
        loaded_files: List of dictionaries from load_all_csv_files

    Returns:
        DataFrame with columns:
            - file_id
            - file_path
            - filename
            - product, sample, direction, run_id (from parsed filename)
            - MeasFreq, Resampling, LowFreq, HighFreq, etc (from metadata)
    """
    rows = []

    for item in loaded_files:
        row = {
            'file_id': item['file_id'],
            'file_path': item['file_path'],
            'filename': item['filename']
        }

        # Add parsed filename info
        row.update(item['parsed_filename'])

        # Add metadata fields
        row.update(item['metadata'])

        rows.append(row)

    metadata_df = pd.DataFrame(rows)

    # Convert numeric metadata to appropriate types
    numeric_cols = ['MeasFreq', 'Resampling', 'LowFreq', 'HighFreq',
                   'xRMS', 'yRMS', 'zRMS', 'sumRMS', 'xPeak', 'yPeak', 'zPeak', 'sumPeak']

    for col in numeric_cols:
        if col in metadata_df.columns:
            metadata_df[col] = pd.to_numeric(metadata_df[col], errors='coerce')

    return metadata_df


def create_combined_timeseries(loaded_files: List[Dict]) -> pd.DataFrame:
    """
    Combine timeseries from all files into one DataFrame

    Args:
        loaded_files: List of dictionaries from load_all_csv_files

    Returns:
        DataFrame with file_id column added to each timeseries
    """
    combined = []

    for item in loaded_files:
        ts = item['timeseries'].copy()
        ts['file_id'] = item['file_id']
        combined.append(ts)

    if not combined:
        return pd.DataFrame()

    combined_df = pd.concat(combined, ignore_index=True)

    # Reorder columns to put file_id first
    cols = ['file_id'] + [c for c in combined_df.columns if c != 'file_id']
    combined_df = combined_df[cols]

    return combined_df
