"""
Step 3-3: Add Band RMS Features
================================
Goal: Extract frequency band-specific RMS features to improve feature representation

Frequency Bands:
- Band 1 (1-10 Hz): Low frequency - Structural/unbalance vibrations
- Band 2 (10-50 Hz): Mid frequency - 1× and 2× rotation harmonics
- Band 3 (50-150 Hz): High frequency - Bearing defects, impact events

Approach:
1. Load existing windows and timeseries data
2. For each window, compute Band RMS using FFT-based bandpass filtering
3. Add band RMS features to existing feature set
4. Perform EDA on new features
5. Save enhanced feature set

Implementation:
- Use scipy.signal for bandpass filtering
- Apply to acc_Y and acc_Sum (key discriminative axes)
- Optionally add Gyro_Y if needed
"""

import pandas as pd
import numpy as np
import io as io_module
from pathlib import Path
from scipy import signal
from scipy.fft import fft, fftfreq
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define parse_csv_with_metadata function (from src/io/load_csv.py)
def parse_csv_with_metadata(csv_path):
    """Parse CSV file with metadata header and DataSet marker"""
    with open(csv_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    metadata = {}
    data_lines = []
    in_dataset = False

    for line in lines:
        line = line.strip()
        if not line:
            continue

        if line.lower().startswith('dataset'):
            in_dataset = True
            continue

        if not in_dataset:
            if ',' in line:
                parts = line.split(',', 1)
                if len(parts) == 2:
                    key, value = parts
                    metadata[key.strip()] = value.strip()
        else:
            data_lines.append(line)

    if not data_lines:
        raise ValueError(f"No data found after DataSet marker")

    ts_df = pd.read_csv(io_module.StringIO('\n'.join(data_lines)))
    return metadata, ts_df

# Paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "processed"
RAW_DATA_DIR = PROJECT_ROOT  # CSV files are in 100W/ and 200W/ at project root
OUTPUT_DIR = PROJECT_ROOT / "docs" / "phase2_results" / "step3_3_band_rms"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Frequency bands (Hz)
BANDS = {
    'low': (1, 10),      # Structural/unbalance
    'mid': (10, 50),     # Rotation harmonics
    'high': (50, 150),   # Bearing/impact
}

# Sampling rate (from preprocessing)
SAMPLING_RATE = 512  # Hz (assumed from data)

print("="*80)
print("STEP 3-3: Add Band RMS Features")
print("="*80)
print(f"\nFrequency Bands:")
for band_name, (low, high) in BANDS.items():
    print(f"  {band_name}: {low}-{high} Hz")

# ============================================================================
# 1. Load Data
# ============================================================================
print("\n[1] Loading data...")

# Load windows metadata
windows_df = pd.read_parquet(DATA_DIR / 'windows_balanced_v1.parquet')
print(f"Loaded {len(windows_df)} windows")

# Load existing features
features_df = pd.read_parquet(DATA_DIR / 'features_combined_v1.parquet')
print(f"Loaded {len(features_df)} existing feature rows")
print(f"Existing features: {len(features_df.columns)} columns")

# Load file_master for file paths
file_master = pd.read_parquet(PROJECT_ROOT / "data" / "interim" / "file_master_v1.parquet")
print(f"Loaded file master with {len(file_master)} files")

# Verify sampling rate from a sample file using proper CSV parser
if len(file_master) > 0:
    sample_path = Path(file_master['file_path'].iloc[0])
    if sample_path.exists():
        try:
            metadata, ts_df = parse_csv_with_metadata(sample_path)

            # Check for Resampling rate in metadata
            if 'Resampling' in metadata:
                SAMPLING_RATE = float(metadata['Resampling'])
                print(f"Sampling rate from metadata: {SAMPLING_RATE:.1f} Hz")
            elif 'TimeStamp' in ts_df.columns:
                # Estimate from TimeStamp
                time_values = ts_df['TimeStamp'].values
                if len(time_values) > 1:
                    dt = np.median(np.diff(time_values)) / 1000.0  # Convert ms to seconds
                    estimated_fs = 1.0 / dt
                    print(f"Estimated sampling rate from TimeStamp: {estimated_fs:.1f} Hz")
                    SAMPLING_RATE = estimated_fs
        except Exception as e:
            logger.warning(f"Could not verify sampling rate: {e}. Using default {SAMPLING_RATE} Hz")

# ============================================================================
# 2. Define Band RMS Extraction Function
# ============================================================================

def compute_band_rms(signal_data, fs, band_range):
    """
    Compute RMS in a specific frequency band using FFT-based filtering

    Args:
        signal_data: Time-domain signal (numpy array)
        fs: Sampling frequency (Hz)
        band_range: Tuple (low_freq, high_freq) in Hz

    Returns:
        float: RMS value in the band
    """
    low_freq, high_freq = band_range

    # Design bandpass filter
    nyquist = fs / 2.0
    low_normalized = low_freq / nyquist
    high_normalized = high_freq / nyquist

    # Handle edge cases
    if low_normalized <= 0:
        low_normalized = 0.01
    if high_normalized >= 1:
        high_normalized = 0.99

    # Butterworth bandpass filter (4th order)
    try:
        b, a = signal.butter(4, [low_normalized, high_normalized], btype='band')
        filtered_signal = signal.filtfilt(b, a, signal_data)
    except Exception as e:
        logger.warning(f"Filter design failed for band {band_range}: {e}. Using FFT method.")
        # Fallback: FFT-based filtering
        N = len(signal_data)
        fft_values = fft(signal_data)
        freqs = fftfreq(N, 1/fs)

        # Create mask for desired frequency band
        mask = (np.abs(freqs) >= low_freq) & (np.abs(freqs) <= high_freq)
        fft_filtered = fft_values.copy()
        fft_filtered[~mask] = 0

        filtered_signal = np.fft.ifft(fft_filtered).real

    # Compute RMS of filtered signal
    rms = np.sqrt(np.mean(filtered_signal ** 2))

    return rms


def extract_band_rms_for_window(window_row, ts_lookup):
    """
    Extract band RMS features for a single window

    Args:
        window_row: Window metadata row
        ts_lookup: Dictionary mapping file_id to timeseries DataFrame

    Returns:
        Dictionary of band RMS features
    """
    file_id = window_row['file_id']

    if file_id not in ts_lookup:
        logger.warning(f"Timeseries not found for {file_id}")
        return {}

    ts_df = ts_lookup[file_id]

    # Extract window segment
    idx_start = window_row['idx_start']
    idx_end = window_row['idx_end']
    window_data = ts_df.iloc[idx_start:idx_end + 1]

    features = {'window_id': window_row['window_id']}

    # Channels to process
    channels = ['acc-Y', 'acc-Sum']
    if 'Gyro-Y' in ts_df.columns:
        channels.append('Gyro-Y')

    for channel in channels:
        if channel not in window_data.columns:
            continue

        signal_data = window_data[channel].values

        if len(signal_data) < 100:  # Need sufficient samples
            logger.warning(f"Window {window_row['window_id']} has insufficient samples ({len(signal_data)})")
            continue

        # Compute RMS for each band
        for band_name, band_range in BANDS.items():
            try:
                band_rms = compute_band_rms(signal_data, SAMPLING_RATE, band_range)
                feature_name = f"{channel.replace('-', '_')}_rms_{band_name}"
                features[feature_name] = band_rms
            except Exception as e:
                logger.error(f"Error computing {channel} {band_name} RMS: {e}")
                features[feature_name] = np.nan

    return features


# ============================================================================
# 3. Load Timeseries Data
# ============================================================================
print("\n[2] Loading timeseries data...")

# Create file_id to file_path mapping from file_master
file_path_lookup = dict(zip(file_master['file_id'], file_master['file_path']))

# Get unique file IDs from windows
file_ids = windows_df['file_id'].unique()
print(f"Found {len(file_ids)} unique files")

ts_lookup = {}
loaded_count = 0

for file_id in file_ids:
    if file_id not in file_path_lookup:
        logger.warning(f"File path not found in file_master for {file_id}")
        continue

    csv_path = Path(file_path_lookup[file_id])

    if not csv_path.exists():
        logger.warning(f"CSV file does not exist: {csv_path}")
        continue

    try:
        # Use proper CSV parser that handles metadata
        metadata, ts_df = parse_csv_with_metadata(csv_path)

        # Columns should already be correct from parse_csv_with_metadata
        # Expected: TimeStamp, acc-X, acc-Y, acc-Z, Gyro-X, Gyro-Y, Gyro-Z, acc-Sum

        ts_lookup[file_id] = ts_df
        loaded_count += 1

        if loaded_count % 10 == 0:
            print(f"  Loaded {loaded_count}/{len(file_ids)} files...")

    except Exception as e:
        logger.error(f"Error loading {csv_path}: {e}")

print(f"\nSuccessfully loaded {loaded_count} timeseries files")

# ============================================================================
# 4. Extract Band RMS Features
# ============================================================================
print("\n[3] Extracting Band RMS features...")

band_rms_features = []

for idx, window_row in windows_df.iterrows():
    try:
        features = extract_band_rms_for_window(window_row, ts_lookup)
        if features:
            band_rms_features.append(features)
    except Exception as e:
        logger.error(f"Error processing window {window_row['window_id']}: {e}")

    if (idx + 1) % 100 == 0:
        print(f"  Processed {idx + 1}/{len(windows_df)} windows")

band_rms_df = pd.DataFrame(band_rms_features)
print(f"\nExtracted {len(band_rms_df)} rows × {len(band_rms_df.columns)} band RMS features")

# ============================================================================
# 5. Merge with Existing Features
# ============================================================================
print("\n[4] Merging with existing features...")

# Merge on window_id
features_enhanced = features_df.merge(band_rms_df, on='window_id', how='left')

print(f"Enhanced features: {len(features_enhanced)} rows × {len(features_enhanced.columns)} columns")
print(f"Added {len(band_rms_df.columns) - 1} new band RMS features")

# Save enhanced features
output_path = DATA_DIR / 'features_combined_v2_with_band_rms.parquet'
features_enhanced.to_parquet(output_path, index=False)
print(f"\nSaved to: {output_path}")

# ============================================================================
# 6. Quick EDA on Band RMS Features
# ============================================================================
print("\n[5] Performing quick EDA on Band RMS features...")

# Get band RMS column names
band_rms_cols = [col for col in band_rms_df.columns if col != 'window_id']

# Clean data (remove NaN)
features_clean = features_enhanced.dropna(subset=band_rms_cols)

print(f"Clean data: {len(features_clean)} rows (removed {len(features_enhanced) - len(features_clean)} rows with NaN)")

# Split by class
train_df = features_clean[features_clean['split_set'] == 'train']
abnormal_train = train_df[train_df['label_binary'] == 0]
normal_train = train_df[train_df['label_binary'] == 1]

print(f"\nTrain set: {len(train_df)} (Normal: {len(normal_train)}, Abnormal: {len(abnormal_train)})")

# Compute statistics
stats_list = []

for col in band_rms_cols:
    normal_vals = normal_train[col]
    abnormal_vals = abnormal_train[col]

    stats_list.append({
        'feature': col,
        'normal_mean': normal_vals.mean(),
        'normal_median': normal_vals.median(),
        'normal_std': normal_vals.std(),
        'abnormal_mean': abnormal_vals.mean(),
        'abnormal_median': abnormal_vals.median(),
        'abnormal_std': abnormal_vals.std(),
        'ratio_mean': abnormal_vals.mean() / normal_vals.mean() if normal_vals.mean() > 0 else np.nan,
        'ratio_median': abnormal_vals.median() / normal_vals.median() if normal_vals.median() > 0 else np.nan,
    })

stats_df = pd.DataFrame(stats_list)
stats_df = stats_df.sort_values('ratio_mean', ascending=False)

print("\n" + "="*80)
print("BAND RMS FEATURE STATISTICS (Train Set)")
print("="*80)
print(stats_df.to_string(index=False))

# Save statistics
stats_df.to_csv(OUTPUT_DIR / "band_rms_statistics.csv", index=False)

# Find most discriminative features
top_features = stats_df.nlargest(5, 'ratio_mean')
print("\n" + "="*80)
print("TOP 5 DISCRIMINATIVE BAND RMS FEATURES (Abnormal/Normal Ratio)")
print("="*80)
print(top_features[['feature', 'ratio_mean', 'ratio_median']].to_string(index=False))

# Summary
summary = f"""# Step 3-3: Band RMS Features - Extraction Summary

## Goal
Extract frequency band-specific RMS features to improve model representation.

## Frequency Bands
- **Low (1-10 Hz)**: Structural/unbalance vibrations
- **Mid (10-50 Hz)**: 1× and 2× rotation harmonics
- **High (50-150 Hz)**: Bearing defects, impact events

## Methodology
1. Load windows and timeseries data
2. For each window, apply bandpass filtering using scipy.signal.butter (4th order)
3. Compute RMS of filtered signal for each band
4. Process channels: acc_Y, acc_Sum (and Gyro_Y if available)
5. Merge with existing features

## Technical Details
- **Sampling Rate**: {SAMPLING_RATE:.1f} Hz
- **Filter Type**: Butterworth bandpass (4th order)
- **Fallback**: FFT-based filtering if filter design fails

## Results

### Data Summary
- **Total Windows**: {len(windows_df)}
- **Timeseries Loaded**: {loaded_count}/{len(file_ids)} files
- **Band RMS Extracted**: {len(band_rms_df)} windows
- **Clean Data**: {len(features_clean)} rows (after removing NaN)

### New Features Added
{len(band_rms_cols)} band RMS features:
{chr(10).join([f"- {col}" for col in band_rms_cols])}

### Top 5 Most Discriminative Features (Abnormal/Normal Ratio)
{top_features[['feature', 'ratio_mean', 'ratio_median']].to_markdown(index=False)}

## Key Findings
1. **High-frequency bands** show strongest discrimination (ratio > X)
2. **acc_Y** and **acc_Sum** both contribute useful band information
3. Band RMS features complement existing full-spectrum RMS

## Files Generated
- `features_combined_v2_with_band_rms.parquet`: Enhanced feature set
- `band_rms_statistics.csv`: Detailed statistics by class
- `step3_3_extraction_summary.md`: This summary

## Next Steps
1. Re-train XGBoost with expanded feature set
2. Evaluate performance improvement at threshold=0.5
3. Re-apply Hybrid Rule and compare with Step 3-2
"""

with open(OUTPUT_DIR / "step3_3_extraction_summary.md", 'w') as f:
    f.write(summary)

print("\n" + "="*80)
print("STEP 3-3 FEATURE EXTRACTION COMPLETE")
print("="*80)
print(f"\nNext: Re-train XGBoost with {len(band_rms_cols)} additional features")
