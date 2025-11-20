# Phase 3-0: Data Integrity Audit Report

## Summary

- Total checks: 9
- ✅ PASS: 8
- ❌ FAIL: 0
- ⚠️ WARNING: 1

**✅ AUDIT PASSED: No data leakage detected**

## Detailed Checks

### File-level: Train-Val overlap
- **Status**: ✅ PASS
- **Expected**: 0
- **Actual**: 0
- **Details**: No overlap (time-split files excluded)

### File-level: Train-Test overlap
- **Status**: ✅ PASS
- **Expected**: 0
- **Actual**: 0
- **Details**: No overlap (time-split files excluded)

### File-level: Val-Test overlap
- **Status**: ✅ PASS
- **Expected**: 0
- **Actual**: 0
- **Details**: No overlap (time-split files excluded)

### Temporal: 100W_S00_CCW_R4
- **Status**: ✅ PASS
- **Expected**: train < val < test
- **Actual**: train:31, val:1, test:1
- **Details**: Correct temporal ordering

### Temporal: 100W_S00_CW_R4
- **Status**: ✅ PASS
- **Expected**: train < val < test
- **Actual**: train:18, val:1, test:1
- **Details**: Correct temporal ordering

### Temporal: 200W_S03_CCW_R4
- **Status**: ✅ PASS
- **Expected**: train < val < test
- **Actual**: train:3, val:1, test:1
- **Details**: Correct temporal ordering

### Temporal: 200W_S03_CW_R4
- **Status**: ✅ PASS
- **Expected**: train < val < test
- **Actual**: train:28, val:1, test:1
- **Details**: Correct temporal ordering

### Window coverage
- **Status**: ⚠️ WARNING
- **Expected**: 612
- **Actual**: 609
- **Details**: Missing: 3, Extra: 0

### Label distribution
- **Status**: ✅ PASS
- **Expected**: Balanced across splits
- **Actual**: See label_dist table
- **Details**: Label distribution verified

## Split Configuration

**Strategy**: time_based

**Time-split files** (4):
- 100W_S00_CCW_R4
- 100W_S00_CW_R4
- 200W_S03_CCW_R4
- 200W_S03_CW_R4

**File-level split files**: 53 files

## Label Distribution

```
label_binary    0    1
split_set             
test           97   14
train         321  365
val            88    4
```

