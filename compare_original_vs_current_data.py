"""
데이터 비교 스크립트: 원본 vs 백업 데이터
사용자가 업로드한 원본 데이터와 백업된 현재 데이터를 비교
"""

import pandas as pd
from pathlib import Path
import hashlib
from collections import defaultdict

PROJECT_ROOT = Path(__file__).parent
ORIGINAL_100W = PROJECT_ROOT / "100W"
ORIGINAL_200W = PROJECT_ROOT / "200W"
BACKUP_100W = PROJECT_ROOT / "data" / "backup_current_20241124" / "100W"
BACKUP_200W = PROJECT_ROOT / "data" / "backup_current_20241124" / "200W"
FILE_MASTER = PROJECT_ROOT / "data" / "interim" / "file_master_v1.parquet"
OUTPUT_REPORT = PROJECT_ROOT / "docs" / "data_comparison_report.md"

def get_file_info(csv_path):
    """파일 정보 추출"""
    return {
        'name': csv_path.name,
        'size': csv_path.stat().st_size,
        'md5': hashlib.md5(csv_path.read_bytes()).hexdigest()[:16]
    }

def compare_directories():
    """100W와 200W 디렉토리 비교"""

    print("="*80)
    print("데이터 비교 분석 시작")
    print("="*80)

    # 1. 파일 목록 수집
    print("\n[1] 파일 목록 수집 중...")

    original_100w_files = sorted(ORIGINAL_100W.glob("*.csv")) if ORIGINAL_100W.exists() else []
    original_200w_files = sorted(ORIGINAL_200W.glob("*.csv")) if ORIGINAL_200W.exists() else []
    backup_100w_files = sorted(BACKUP_100W.glob("*.csv"))
    backup_200w_files = sorted(BACKUP_200W.glob("*.csv"))

    print(f"  원본 100W: {len(original_100w_files)} 파일")
    print(f"  원본 200W: {len(original_200w_files)} 파일")
    print(f"  백업 100W: {len(backup_100w_files)} 파일")
    print(f"  백업 200W: {len(backup_200w_files)} 파일")

    # 2. file_master 로드
    print("\n[2] file_master.parquet 분석 중...")
    fm = pd.read_parquet(FILE_MASTER)
    print(f"  file_master 레코드: {len(fm)} 개")
    print(f"  is_usable=True: {fm['is_usable'].sum()} 개")

    # 3. 파일명 비교
    print("\n[3] 파일명 비교 중...")

    original_100w_names = {f.name for f in original_100w_files}
    original_200w_names = {f.name for f in original_200w_files}
    backup_100w_names = {f.name for f in backup_100w_files}
    backup_200w_names = {f.name for f in backup_200w_files}

    # 100W 비교
    added_100w = original_100w_names - backup_100w_names
    removed_100w = backup_100w_names - original_100w_names
    common_100w = original_100w_names & backup_100w_names

    # 200W 비교
    added_200w = original_200w_names - backup_200w_names
    removed_200w = backup_200w_names - original_200w_names
    common_200w = original_200w_names & backup_200w_names

    print(f"\n  100W 비교:")
    print(f"    추가된 파일: {len(added_100w)} 개")
    print(f"    제거된 파일: {len(removed_100w)} 개")
    print(f"    공통 파일: {len(common_100w)} 개")

    print(f"\n  200W 비교:")
    print(f"    추가된 파일: {len(added_200w)} 개")
    print(f"    제거된 파일: {len(removed_200w)} 개")
    print(f"    공통 파일: {len(common_200w)} 개")

    # 4. Sample00, 01 확인
    print("\n[4] Sample00, 01 파일 확인...")

    sample00_01_original = [f for f in original_100w_names if 'Sample00' in f or 'Sample01' in f]
    sample00_01_backup = [f for f in backup_100w_names if 'Sample00' in f or 'Sample01' in f]

    print(f"  원본에 Sample00/01: {len(sample00_01_original)} 개")
    if sample00_01_original:
        for f in sorted(sample00_01_original):
            print(f"    - {f}")

    print(f"  백업에 Sample00/01: {len(sample00_01_backup)} 개")

    # 5. file_master와 매칭
    print("\n[5] file_master와 파일 매칭 분석...")

    fm_filenames = set(fm['filename'].tolist())
    original_all_names = original_100w_names | original_200w_names

    fm_matched = fm_filenames & original_all_names
    fm_missing = fm_filenames - original_all_names
    extra_files = original_all_names - fm_filenames

    print(f"  file_master 매칭: {len(fm_matched)} / {len(fm_filenames)} 개")
    print(f"  file_master에만 있음: {len(fm_missing)} 개")
    print(f"  원본에만 있음: {len(extra_files)} 개")

    # 6. 보고서 생성
    print("\n[6] 보고서 생성 중...")

    report_lines = []
    report_lines.append("# 데이터 비교 분석 보고서")
    report_lines.append(f"\n**분석 일시**: {pd.Timestamp.now()}")
    report_lines.append("\n---\n")

    report_lines.append("## 1. 파일 개수 요약\n")
    report_lines.append("| 위치 | 100W | 200W | 총계 |")
    report_lines.append("|------|------|------|------|")
    report_lines.append(f"| **원본 (업로드)** | {len(original_100w_files)} | {len(original_200w_files)} | {len(original_100w_files) + len(original_200w_files)} |")
    report_lines.append(f"| **백업 (2024-11-24)** | {len(backup_100w_files)} | {len(backup_200w_files)} | {len(backup_100w_files) + len(backup_200w_files)} |")
    report_lines.append(f"| **file_master 기록** | - | - | {len(fm)} ({fm['is_usable'].sum()} usable) |")

    report_lines.append("\n## 2. 파일 변경 사항\n")
    report_lines.append("### 100W 디렉토리\n")
    report_lines.append(f"- **추가된 파일**: {len(added_100w)} 개")
    if added_100w:
        report_lines.append("\n```")
        for f in sorted(added_100w):
            report_lines.append(f)
        report_lines.append("```\n")

    report_lines.append(f"\n- **제거된 파일**: {len(removed_100w)} 개")
    if removed_100w:
        report_lines.append("\n```")
        for f in sorted(removed_100w):
            report_lines.append(f)
        report_lines.append("```\n")

    report_lines.append("\n### 200W 디렉토리\n")
    report_lines.append(f"- **추가된 파일**: {len(added_200w)} 개")
    if added_200w:
        report_lines.append("\n```")
        for f in sorted(added_200w):
            report_lines.append(f)
        report_lines.append("```\n")

    report_lines.append(f"\n- **제거된 파일**: {len(removed_200w)} 개")
    if removed_200w:
        report_lines.append("\n```")
        for f in sorted(removed_200w):
            report_lines.append(f)
        report_lines.append("```\n")

    report_lines.append("\n## 3. Sample00/01 (Normal 파일) 확인\n")
    report_lines.append(f"- **원본에 Sample00/01**: {len(sample00_01_original)} 개")
    if sample00_01_original:
        report_lines.append("\n```")
        for f in sorted(sample00_01_original):
            report_lines.append(f)
        report_lines.append("```\n")
    else:
        report_lines.append("\n**⚠️ 원본에도 Sample00/01 파일이 없습니다!**\n")

    report_lines.append(f"\n- **백업에 Sample00/01**: {len(sample00_01_backup)} 개")

    report_lines.append("\n## 4. file_master.parquet 매칭 분석\n")
    report_lines.append(f"- **file_master 총 레코드**: {len(fm)} 개")
    report_lines.append(f"- **원본 파일과 매칭**: {len(fm_matched)} 개")
    report_lines.append(f"- **file_master에만 존재** (누락): {len(fm_missing)} 개")
    report_lines.append(f"- **원본에만 존재** (추가): {len(extra_files)} 개")

    if fm_missing:
        report_lines.append("\n### file_master에는 있으나 원본에 없는 파일 (누락)\n")
        report_lines.append("```")
        for f in sorted(fm_missing)[:20]:
            report_lines.append(f)
        if len(fm_missing) > 20:
            report_lines.append(f"... (총 {len(fm_missing)}개 중 20개만 표시)")
        report_lines.append("```\n")

    if extra_files:
        report_lines.append("\n### 원본에는 있으나 file_master에 없는 파일 (추가)\n")
        report_lines.append("```")
        for f in sorted(extra_files)[:20]:
            report_lines.append(f)
        if len(extra_files) > 20:
            report_lines.append(f"... (총 {len(extra_files)}개 중 20개만 표시)")
        report_lines.append("```\n")

    report_lines.append("\n## 5. 결론 및 권장사항\n")

    total_original = len(original_100w_files) + len(original_200w_files)
    total_backup = len(backup_100w_files) + len(backup_200w_files)

    if total_original > total_backup:
        report_lines.append(f"✅ **원본 데이터 복구 성공**: {total_backup}개 → {total_original}개 (+{total_original - total_backup}개)")

        if len(sample00_01_original) > 0:
            report_lines.append(f"\n✅ **Sample00/01 파일 복구됨**: {len(sample00_01_original)}개")
            report_lines.append("\n**다음 단계**:")
            report_lines.append("1. 전체 {total_original}개 파일로 전처리 파이프라인 재실행")
            report_lines.append("2. Phase3 parquet 재생성 (898 windows 확보 가능 여부 확인)")
            report_lines.append("3. ECMiner Stage1 구현 진행")
        else:
            report_lines.append("\n⚠️ **Sample00/01 파일은 여전히 누락**")
            report_lines.append("\n**다음 단계**:")
            report_lines.append("1. Sample00/01 파일 추가 확인 필요")
            report_lines.append("2. 또는 Phase3 parquet 활용 방안 진행")

    elif total_original == total_backup:
        report_lines.append(f"⚠️ **파일 개수 동일**: {total_original}개")
        report_lines.append("\n원본 데이터가 백업과 동일합니다. 파일이 복구되지 않았습니다.")
        report_lines.append("\n**권장사항**: Phase3 parquet를 활용한 ECMiner 구현 진행")

    else:
        report_lines.append(f"❌ **파일 감소**: {total_backup}개 → {total_original}개 (-{total_backup - total_original}개)")
        report_lines.append("\n백업보다 파일이 적습니다. 데이터 확인이 필요합니다.")

    # 보고서 저장
    OUTPUT_REPORT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_REPORT.write_text("\n".join(report_lines), encoding='utf-8')

    print(f"\n✓ 보고서 저장: {OUTPUT_REPORT}")
    print("\n" + "="*80)
    print("분석 완료!")
    print("="*80)

    return {
        'original_total': total_original,
        'backup_total': total_backup,
        'added_files': len(added_100w) + len(added_200w),
        'removed_files': len(removed_100w) + len(removed_200w),
        'sample00_01_found': len(sample00_01_original) > 0
    }

if __name__ == '__main__':
    result = compare_directories()

    print("\n요약:")
    print(f"  원본: {result['original_total']} 파일")
    print(f"  백업: {result['backup_total']} 파일")
    print(f"  추가: +{result['added_files']} 파일")
    print(f"  제거: -{result['removed_files']} 파일")
    print(f"  Sample00/01: {'✓ 발견' if result['sample00_01_found'] else '✗ 누락'}")
