"""
데이터 누수(Data Leakage) 검증 스크립트

데이터 분할 원칙 검증:
1. 동일 파일의 윈도우가 train/val/test에 분산되지 않았는지 확인
2. 시간 기반 분할 시 윈도우가 경계를 넘지 않는지 확인
3. CW/CCW 방향이 같은 split에 유지되는지 확인
4. Balancing 시 원본 데이터의 독립성이 유지되는지 확인
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json

def verify_no_file_leakage(windows_df: pd.DataFrame, file_master_df: pd.DataFrame) -> dict:
    """
    검증 1: 동일 파일이 여러 split에 분산되지 않았는지 확인

    원칙: 한 파일의 윈도우는 모두 같은 split에 속해야 함
          (time_split 제외 - 이는 의도적으로 분할됨)
    """
    print("=" * 80)
    print("검증 1: 파일 단위 데이터 누수 확인")
    print("=" * 80)

    issues = []

    # time_split 파일 목록 가져오기
    time_split_files = set(
        file_master_df[file_master_df['split_set'] == 'time_split']['file_id'].values
    )

    # 파일별로 어느 split에 속하는지 확인
    file_splits = windows_df.groupby('file_id')['split_set'].unique()

    for file_id, splits in file_splits.items():
        if len(splits) > 1:
            # time_split 파일은 예외 (의도적으로 분할됨)
            if file_id not in time_split_files:
                issues.append({
                    'file_id': file_id,
                    'splits': list(splits),
                    'issue': 'File appears in multiple splits (NOT time_split)'
                })
            else:
                # time_split 파일은 정상적으로 분할됨
                print(f"ℹ️  {file_id}: time-based split → {list(splits)} (정상)")

    if issues:
        print(f"\n❌ 발견된 문제: {len(issues)}개")
        for issue in issues:
            print(f"  - {issue['file_id']}: {issue['splits']}")
    else:
        print("\n✅ 통과: 모든 비time_split 파일이 단일 split에 속함")

    return {
        'passed': len(issues) == 0,
        'issues': issues
    }


def verify_time_split_boundaries(windows_df: pd.DataFrame, file_master_df: pd.DataFrame) -> dict:
    """
    검증 2: 시간 기반 분할 윈도우의 경계 준수 확인

    원칙: time_split 파일의 윈도우는 지정된 시간 범위 내에만 있어야 함
          (train: 0-60%, val: 60-80%, test: 80-100%)
    """
    print("\n" + "=" * 80)
    print("검증 2: 시간 기반 분할 경계 준수 확인")
    print("=" * 80)

    # time_split 파일 목록
    time_split_files = file_master_df[
        file_master_df['split_set'] == 'time_split'
    ]['file_id'].unique()

    time_split_windows = windows_df[
        windows_df['file_id'].isin(time_split_files)
    ].copy()

    if len(time_split_windows) == 0:
        print("⚠️  time_split 윈도우가 없음")
        return {'passed': True, 'issues': []}

    issues = []

    # 각 time_split 파일별로 검증
    for file_id in time_split_windows['file_id'].unique():
        file_windows = time_split_windows[time_split_windows['file_id'] == file_id]

        # 각 split별로 시간 범위 확인
        for split_name in ['train', 'val', 'test']:
            split_windows = file_windows[file_windows['split_set'] == split_name]

            if len(split_windows) == 0:
                continue

            # 윈도우 시작/끝 시간 확인
            start_times = split_windows['start_time'].values
            end_times = split_windows['end_time'].values

            # 예상 범위
            if split_name == 'train':
                expected_range = (0.0, 0.6)
            elif split_name == 'val':
                expected_range = (0.6, 0.8)
            else:  # test
                expected_range = (0.8, 1.0)

            # 파일 길이 추정
            max_end = file_windows['end_time'].max()
            expected_start_abs = max_end * expected_range[0]
            expected_end_abs = max_end * expected_range[1]

            # 경계 위반 확인 (윈도우 시작 시간의 비율 기준)
            # 모든 윈도우의 시작 시간이 예상 범위 내에 있어야 함
            start_ratios = start_times / max_end

            # 범위를 다소 완화 (윈도우 길이 8초를 고려)
            tolerance = 0.05  # ±5% 허용

            boundary_violations = (
                (start_ratios < expected_range[0] - tolerance) |
                (start_ratios > expected_range[1] + tolerance)
            )

            if boundary_violations.any():
                actual_start_ratio = start_ratios.min()
                actual_end_ratio = start_ratios.max()
                issues.append({
                    'file_id': file_id,
                    'split': split_name,
                    'expected_range': expected_range,
                    'actual_range': (f'{actual_start_ratio:.1%}', f'{actual_end_ratio:.1%}'),
                    'violations': boundary_violations.sum()
                })

    if issues:
        print(f"❌ 발견된 문제: {len(issues)}개")
        for issue in issues:
            print(f"  - {issue['file_id']} ({issue['split']}): "
                  f"expected {issue['expected_range']}, "
                  f"got {issue['actual_range']}")
    else:
        print("✅ 통과: 모든 time_split 윈도우가 올바른 시간 범위 내에 있음")

    return {
        'passed': len(issues) == 0,
        'issues': issues
    }


def verify_direction_consistency(windows_df: pd.DataFrame) -> dict:
    """
    검증 3: CW/CCW 방향이 같은 파일 내에서 일관되게 유지되는지 확인

    원칙: 한 파일의 모든 윈도우는 같은 방향(CW 또는 CCW)을 가져야 함
    """
    print("\n" + "=" * 80)
    print("검증 3: CW/CCW 방향 일관성 확인")
    print("=" * 80)

    issues = []

    # 파일별 방향 확인
    file_directions = windows_df.groupby('file_id')['direction'].unique()

    for file_id, directions in file_directions.items():
        if len(directions) > 1:
            issues.append({
                'file_id': file_id,
                'directions': list(directions),
                'issue': 'Multiple directions in same file'
            })

    if issues:
        print(f"❌ 발견된 문제: {len(issues)}개")
        for issue in issues:
            print(f"  - {issue['file_id']}: {issue['directions']}")
    else:
        print("✅ 통과: 모든 파일이 단일 방향을 유지함")

    return {
        'passed': len(issues) == 0,
        'issues': issues
    }


def verify_balancing_independence(
    windows_original: pd.DataFrame,
    windows_balanced: pd.DataFrame
) -> dict:
    """
    검증 4: Balancing 후 데이터 독립성 확인

    원칙:
    - Balancing은 train set에만 적용
    - Val/Test는 원본 그대로 유지
    - Oversampling 시 복제된 샘플이 명확히 표시됨
    """
    print("\n" + "=" * 80)
    print("검증 4: Balancing 후 데이터 독립성 확인")
    print("=" * 80)

    issues = []

    # Val/Test 윈도우 수 확인
    for split in ['val', 'test']:
        orig_count = len(windows_original[windows_original['split_set'] == split])
        balanced_count = len(windows_balanced[windows_balanced['split_set'] == split])

        if orig_count != balanced_count:
            issues.append({
                'split': split,
                'original': orig_count,
                'balanced': balanced_count,
                'issue': f'{split} set modified during balancing'
            })
        else:
            print(f"✅ {split.upper()}: {orig_count} → {balanced_count} (변화 없음)")

    # Train set 변화 확인
    train_orig = len(windows_original[windows_original['split_set'] == 'train'])
    train_balanced = len(windows_balanced[windows_balanced['split_set'] == 'train'])

    print(f"ℹ️  TRAIN: {train_orig} → {train_balanced} "
          f"({'증가' if train_balanced > train_orig else '감소'})")

    # 중복 윈도우 확인 (train에서만)
    train_windows = windows_balanced[windows_balanced['split_set'] == 'train']
    duplicate_count = len(train_windows) - train_windows['window_id'].nunique()

    if duplicate_count > 0:
        print(f"ℹ️  Oversampling으로 인한 중복: {duplicate_count}개")

    if issues:
        print(f"\n❌ 발견된 문제: {len(issues)}개")
        for issue in issues:
            print(f"  - {issue['split']}: {issue['original']} → {issue['balanced']}")
    else:
        print("\n✅ 통과: Val/Test는 변경되지 않음, Train만 balancing 적용")

    return {
        'passed': len(issues) == 0,
        'issues': issues,
        'train_change': train_balanced - train_orig,
        'duplicates': duplicate_count
    }


def verify_window_overlap_within_split(windows_df: pd.DataFrame, file_master_df: pd.DataFrame) -> dict:
    """
    검증 5: Split 내 윈도우 중첩 확인

    원칙: 같은 split 내에서는 윈도우가 중첩될 수 있음 (50% overlap)
          하지만 다른 split과는 시간적으로 겹치면 안 됨
    """
    print("\n" + "=" * 80)
    print("검증 5: Split 간 윈도우 중첩 확인")
    print("=" * 80)

    issues = []

    # time_split 파일만 확인 (일반 파일은 파일 단위로 분할되므로 겹칠 수 없음)
    time_split_files = file_master_df[
        file_master_df['split_set'] == 'time_split'
    ]['file_id'].unique()

    for file_id in time_split_files:
        file_windows = windows_df[windows_df['file_id'] == file_id]

        # Train, Val, Test 윈도우 시간 범위
        splits_ranges = {}
        for split_name in ['train', 'val', 'test']:
            split_windows = file_windows[file_windows['split_set'] == split_name]
            if len(split_windows) > 0:
                splits_ranges[split_name] = {
                    'min_start': split_windows['start_time'].min(),
                    'max_end': split_windows['end_time'].max()
                }

        # Split 간 중첩 확인
        for split1, range1 in splits_ranges.items():
            for split2, range2 in splits_ranges.items():
                if split1 >= split2:
                    continue

                # 시간 범위 겹침 확인
                overlap = (
                    range1['min_start'] < range2['max_end'] and
                    range2['min_start'] < range1['max_end']
                )

                if overlap:
                    issues.append({
                        'file_id': file_id,
                        'split1': split1,
                        'split2': split2,
                        'range1': (range1['min_start'], range1['max_end']),
                        'range2': (range2['min_start'], range2['max_end'])
                    })

    if issues:
        print(f"❌ 발견된 문제: {len(issues)}개")
        for issue in issues:
            print(f"  - {issue['file_id']}: {issue['split1']} ↔ {issue['split2']} 시간 중첩")
    else:
        print("✅ 통과: Split 간 시간 범위 중첩 없음")

    return {
        'passed': len(issues) == 0,
        'issues': issues
    }


def main():
    print("\n")
    print("╔" + "=" * 78 + "╗")
    print("║" + " " * 20 + "데이터 누수(Data Leakage) 검증" + " " * 28 + "║")
    print("╚" + "=" * 78 + "╝")
    print()

    # Load data
    base_path = Path(__file__).parent / "data"

    windows_original = pd.read_parquet(base_path / "interim" / "windows_meta_v1.parquet")
    windows_balanced = pd.read_parquet(base_path / "processed" / "windows_balanced_v1.parquet")
    file_master = pd.read_parquet(base_path / "interim" / "file_master_v1.parquet")

    print(f"데이터 로딩 완료:")
    print(f"  - Original windows: {len(windows_original)}")
    print(f"  - Balanced windows: {len(windows_balanced)}")
    print(f"  - File master: {len(file_master)} files")
    print()

    # Run all verifications
    results = {}

    results['file_leakage'] = verify_no_file_leakage(windows_balanced, file_master)
    results['time_boundaries'] = verify_time_split_boundaries(windows_balanced, file_master)
    results['direction_consistency'] = verify_direction_consistency(windows_balanced)
    results['balancing_independence'] = verify_balancing_independence(
        windows_original, windows_balanced
    )
    results['window_overlap'] = verify_window_overlap_within_split(windows_balanced, file_master)

    # Final summary
    print("\n" + "=" * 80)
    print("최종 검증 결과")
    print("=" * 80)

    all_passed = all(result['passed'] for result in results.values())

    for test_name, result in results.items():
        status = "✅ 통과" if result['passed'] else "❌ 실패"
        print(f"{test_name:30s}: {status}")

    print("\n" + "=" * 80)
    if all_passed:
        print("🎉 모든 검증 통과! 데이터 누수 없음")
    else:
        print("⚠️  일부 검증 실패 - 위의 문제 내역 확인 필요")
    print("=" * 80)

    return results


if __name__ == "__main__":
    results = main()
