#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
레이블 Excel → CSV 변환 스크립트

"시험전 시료 표기내용.xlsx" 파일을 읽어서 ecminer_labels.csv 생성

사용법:
    python extract_labels_from_excel.py

입력:
    - 시험전 시료 표기내용.xlsx (같은 폴더 또는 부모 폴더)

출력:
    - ecminer_labels.csv (같은 폴더에 생성)
    - 형식: product,sample,label
"""

import sys
from pathlib import Path
import pandas as pd

def find_excel_file(script_dir: Path) -> Path:
    """
    레이블 Excel 파일 찾기

    우선순위:
    1. 같은 폴더
    2. 부모 폴더
    """
    candidates = [
        script_dir / "시험전 시료 표기내용.xlsx",
        script_dir.parent / "시험전 시료 표기내용.xlsx"
    ]

    for path in candidates:
        if path.exists():
            return path

    raise FileNotFoundError(
        f"레이블 Excel 파일을 찾을 수 없습니다.\n"
        f"다음 위치 중 하나에 '시험전 시료 표기내용.xlsx' 파일을 배치하세요:\n"
        f"  1. {script_dir}\n"
        f"  2. {script_dir.parent}"
    )

def parse_excel_labels(excel_path: Path) -> pd.DataFrame:
    """
    Excel 파일에서 레이블 정보 추출

    Excel 구조:
    Row 0: 제목 (시험전 시료 표기내용)
    Row 1: 빈 행
    Row 2: 헤더 (시료번호, 100W, 200W)
    Row 3+: 데이터 (sample, 100W label, 200W label)

    Args:
        excel_path: Excel 파일 경로

    Returns:
        DataFrame with columns: product, sample, label
    """
    # Excel 읽기 (헤더 없이)
    df = pd.read_excel(excel_path, header=None)

    # 데이터 시작 행 찾기 (Row 2: "시료번호" 포함)
    header_row = None
    for idx, row in df.iterrows():
        if '시료번호' in str(row.values):
            header_row = idx
            break

    if header_row is None:
        raise ValueError("Excel 파일에서 '시료번호' 헤더를 찾을 수 없습니다.")

    # 헤더 추출
    headers = df.iloc[header_row].tolist()

    # 데이터 추출 (헤더 다음 행부터)
    data_start = header_row + 1
    data_df = df.iloc[data_start:].reset_index(drop=True)

    # 컬럼 인덱스 찾기
    sample_col = None
    product_cols = {}

    for idx, header in enumerate(headers):
        header_str = str(header).strip()
        if '시료번호' in header_str or header_str == 'sample':
            sample_col = idx
        elif 'W' in header_str and header_str != 'nan':
            product_cols[header_str] = idx

    if sample_col is None:
        raise ValueError("시료번호 컬럼을 찾을 수 없습니다.")

    if not product_cols:
        raise ValueError("제품 컬럼(100W, 200W)을 찾을 수 없습니다.")

    # 레이블 데이터 추출
    labels = []

    for _, row in data_df.iterrows():
        sample = row.iloc[sample_col]

        # 샘플 번호가 유효한지 확인
        if pd.isna(sample):
            continue

        try:
            sample = int(sample)
        except (ValueError, TypeError):
            continue

        # 각 제품별 레이블 추출
        for product, col_idx in product_cols.items():
            label = row.iloc[col_idx]

            # NaN이 아닌 유효한 레이블만 추가
            if pd.notna(label):
                label_str = str(label).strip()
                if label_str and label_str != 'nan':
                    labels.append({
                        'product': product,
                        'sample': sample,
                        'label': label_str
                    })

    # DataFrame 생성
    labels_df = pd.DataFrame(labels)

    # 정렬 (product, sample 순)
    labels_df = labels_df.sort_values(['product', 'sample']).reset_index(drop=True)

    return labels_df

def main():
    """메인 실행 함수"""
    print("=" * 60)
    print("레이블 Excel → CSV 변환 스크립트")
    print("=" * 60)

    # 스크립트 경로
    script_dir = Path(__file__).parent if '__file__' in globals() else Path('.')

    # Excel 파일 찾기
    try:
        excel_path = find_excel_file(script_dir)
        print(f"\n✓ Excel 파일 발견: {excel_path.name}")
    except FileNotFoundError as e:
        print(f"\n✗ 오류: {e}")
        sys.exit(1)

    # Excel 파싱
    try:
        labels_df = parse_excel_labels(excel_path)
        print(f"✓ 레이블 추출 완료: {len(labels_df)}개")
    except Exception as e:
        print(f"\n✗ Excel 파싱 오류: {e}")
        sys.exit(1)

    # 레이블 분포 출력
    print("\n레이블 분포:")
    for product in labels_df['product'].unique():
        product_df = labels_df[labels_df['product'] == product]
        print(f"\n  {product}:")
        label_counts = product_df['label'].value_counts()
        for label, count in label_counts.items():
            print(f"    - {label}: {count}개")

    # CSV 저장
    output_path = script_dir / 'ecminer_labels.csv'
    labels_df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"\n✓ CSV 저장 완료: {output_path.name}")

    # 미리보기
    print("\n생성된 CSV 미리보기 (처음 10개):")
    print(labels_df.head(10).to_string(index=False))

    print("\n" + "=" * 60)
    print("변환 완료!")
    print("=" * 60)

if __name__ == "__main__":
    main()
