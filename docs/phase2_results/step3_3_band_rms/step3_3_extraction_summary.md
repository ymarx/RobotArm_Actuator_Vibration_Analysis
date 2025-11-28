# Step 3-3: Band RMS Features - Extraction Summary

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
- **Sampling Rate**: 512.0 Hz
- **Filter Type**: Butterworth bandpass (4th order)
- **Fallback**: FFT-based filtering if filter design fails

## Results

### Data Summary
- **Total Windows**: 650
- **Timeseries Loaded**: 57/57 files
- **Band RMS Extracted**: 650 windows
- **Clean Data**: 647 rows (after removing NaN)

### New Features Added
9 band RMS features:
- acc_Y_rms_low
- acc_Y_rms_mid
- acc_Y_rms_high
- acc_Sum_rms_low
- acc_Sum_rms_mid
- acc_Sum_rms_high
- Gyro_Y_rms_low
- Gyro_Y_rms_mid
- Gyro_Y_rms_high

### Top 5 Most Discriminative Features (Abnormal/Normal Ratio)
| feature         |   ratio_mean |   ratio_median |
|:----------------|-------------:|---------------:|
| acc_Y_rms_high  |      2.70196 |        1.15186 |
| Gyro_Y_rms_mid  |      2.17151 |        1.09593 |
| Gyro_Y_rms_low  |      2.16505 |        1.15812 |
| Gyro_Y_rms_high |      2.13443 |        1.14584 |
| acc_Y_rms_low   |      1.72132 |        1.13411 |

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
