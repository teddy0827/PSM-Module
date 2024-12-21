import pandas as pd
import numpy as np

def read_csv_data(file_path: str) -> pd.DataFrame:
    """
    CSV 파일을 읽어서 DataFrame으로 반환한다.
    """
    return pd.read_csv(file_path)

def calculate_radius_and_region(df: pd.DataFrame) -> pd.DataFrame:
    """
    웨이퍼 좌표(wf_x, wf_y)를 이용해 반지름을 구하고,
    반지름 구간에 따라 region을 부여한다. (Outside 구간 제외)
    """
    df['radius'] = np.sqrt(df['wf_x']**2 + df['wf_y']**2)
    conditions = [
        (df['radius'] <= 50000),
        (df['radius'] > 50000) & (df['radius'] <= 100000),
        (df['radius'] > 100000)
    ]
    choices = ['Center', 'Middle', 'Edge']
    df['region'] = np.select(conditions, choices)
    return df



def calculate_pershot_m3s(df: pd.DataFrame, group_columns: list, columns_to_calculate: list) -> pd.DataFrame:
    """
    그룹별로 지정된 컬럼의 |mean| + 3 * std를 계산하고 추가 정보를 병합한다.
    """
    grouped = df.groupby(group_columns)[columns_to_calculate].agg(['mean', 'std']).reset_index()

    # (1) MultiIndex -> 단일 레벨(flatten)으로 변환
    #   [('X_reg', 'mean'), ('X_reg', 'std'), ('Y_reg', 'mean'), ('Y_reg', 'std') ...]
    #   ->  ['X_reg_mean', 'X_reg_std', 'Y_reg_mean', 'Y_reg_std' ...]
    grouped.columns = [
        '_'.join([str(c) for c in col if c])  # join할 때 빈 문자열('')이 들어있다면 제외
        if isinstance(col, tuple) else col
        for col in grouped.columns
    ]

    # 혹은 더 간단히, pandas >= 1.5 부터 지원되는 to_flat_index() 사용:
    # grouped.columns = ['_'.join(map(str, col)).rstrip('_') for col in grouped.columns.to_flat_index()]

    # (2) 그룹별 첫 번째 행 추가 정보 병합
    additional_columns = [
        'STEPSEQ', 'LOT_ID', 'Wafer', 'P_EQPID', 'Photo_PPID',
        'P_TIME', 'M_TIME', 'ChuckID', 'ReticleID',
        'Base_EQP1', 'GROUP', 'MMO_MRC_EQP'
    ]

    # 중복 제거
    additional_info = df[group_columns + additional_columns].drop_duplicates(subset=group_columns).copy()

    # (3) merge 시에는 group_columns가 평탄화된 상태와 동일해야 함
    #     하지만 group_columns는 ['UNIQUE_ID', 'DieX', 'DieY'] 같은 단순 리스트이므로 평탄화 불필요
    result = grouped.merge(additional_info, on=group_columns, how='left')

    # (4) |mean| + 3*std 컬럼 생성
    for col in columns_to_calculate:
        mean_col = f'{col}_mean'
        std_col = f'{col}_std'
        if mean_col in result.columns and std_col in result.columns:
            result[f'{col}_abs_m3s'] = result[mean_col].abs() + 3 * result[std_col]


    # 컬럼 순서 조정
    desired_order = (
        group_columns +
        additional_columns +
        [col for col in result.columns if col not in group_columns + additional_columns]
    )
    result = result[desired_order]


    return result


def group_by_region_and_calculate(df: pd.DataFrame, columns_to_calculate: list) -> pd.DataFrame:
    """
    UNIQUE_ID, region 별로 그룹화하여,
    mean, std, mean의 abs, mean+3*sigma 계산을 수행한 결과를 반환한다.
    그룹별 첫 번째 행의 정보를 함께 병합한다.
    """
    grouped = df.groupby(['UNIQUE_ID', 'region']).agg({col: ['mean', 'std'] for col in columns_to_calculate})
    grouped.columns = [f'{col}_{stat}' for col, stat in grouped.columns]

    for col in columns_to_calculate:
        grouped[f'{col}_mean'] = grouped[f'{col}_mean'].abs()
        grouped[f'{col}_m3s'] = grouped[f'{col}_mean'] + 3 * grouped[f'{col}_std']

    first_values = df.groupby(['UNIQUE_ID', 'region']).first().reset_index()
    grouped = grouped.reset_index()
    grouped = grouped.merge(
        first_values[[
            'UNIQUE_ID', 'region', 'STEPSEQ', 'LOT_ID', 'Wafer', 'P_EQPID',
            'Photo_PPID', 'P_TIME', 'M_TIME', 'ChuckID', 'ReticleID',
            'Base_EQP1', 'GROUP', 'MMO_MRC_EQP'
        ]],
        on=['UNIQUE_ID', 'region'],
        how='left'
    )

    return grouped

def group_all_and_calculate(df: pd.DataFrame, columns_to_calculate: list) -> pd.DataFrame:
    """
    UNIQUE_ID 별로 그룹화하여,
    mean, std, mean의 abs, mean+3*sigma 계산을 수행한 결과를 반환한다.
    그룹별 첫 번째 행의 정보를 함께 병합한다.
    """
    grouped_all = df.groupby('UNIQUE_ID').agg({col: ['mean', 'std'] for col in columns_to_calculate})
    grouped_all.columns = [f'{col}_{stat}' for col, stat in grouped_all.columns]

    for col in columns_to_calculate:
        grouped_all[f'{col}_mean'] = grouped_all[f'{col}_mean'].abs()
        grouped_all[f'{col}_m3s'] = grouped_all[f'{col}_mean'] + 3 * grouped_all[f'{col}_std']

    first_values_all = df.groupby('UNIQUE_ID').first().reset_index()
    grouped_all = grouped_all.reset_index()
    grouped_all = grouped_all.merge(
        first_values_all[[
            'UNIQUE_ID', 'STEPSEQ', 'LOT_ID', 'Wafer', 'P_EQPID',
            'Photo_PPID', 'P_TIME', 'M_TIME', 'ChuckID', 'ReticleID',
            'Base_EQP1', 'GROUP', 'MMO_MRC_EQP'
        ]],
        on='UNIQUE_ID',
        how='left'
    )

    grouped_all['region'] = 'All'

    return grouped_all

def reorder_columns_and_concatenate(grouped_by_region: pd.DataFrame,
                                    grouped_all: pd.DataFrame,
                                    columns_to_calculate: list) -> pd.DataFrame:
    """
    region별 그룹화 결과와 전체 그룹화 결과를 병합하고,
    원하는 컬럼 순서로 재정렬하여 반환한다.
    """
    grouped_combined = pd.concat([grouped_by_region, grouped_all], ignore_index=True)

    desired_column_order = [
        'UNIQUE_ID', 'region', 'STEPSEQ', 'LOT_ID', 'Wafer', 'P_EQPID',
        'Photo_PPID', 'P_TIME', 'M_TIME', 'ChuckID', 'ReticleID', 'Base_EQP1',
        'MMO_MRC_EQP', 'GROUP'
    ]

    # 계산된 컬럼만 순서에 추가 (존재하는 경우에만 추가되도록 수정)
    for col in columns_to_calculate:
        optional_columns = [
            f'{col}_mean',
            f'{col}_std',
            f'{col}_m3s'
        ]
        desired_column_order.extend([c for c in optional_columns if c in grouped_combined.columns])

    grouped_combined = grouped_combined[desired_column_order]

    return grouped_combined


def process_data(input_file_path: str, shot_output_file_path: str, all_output_file_path: str):
    df = read_csv_data(input_file_path)
    df = calculate_radius_and_region(df)
    columns_to_calculate = [
        'X_reg', 'Y_reg', 'pred_x', 'pred_y', 'residual_x', 'residual_y', 'psm_fit_x', 'psm_fit_y',
        'residual_x_depsm', 'residual_y_depsm', 'cpe19p_pred_x', 'cpe19p_pred_y', 'cpe19p_resi_x', 'cpe19p_resi_y',
        'ideal_psm_x', 'ideal_psm_y', 'delta_psm_x', 'delta_psm_y'
    ]
    grouped_by_region = group_by_region_and_calculate(df, columns_to_calculate)
    grouped_all = group_all_and_calculate(df, columns_to_calculate)
    shot_m3s = calculate_pershot_m3s(df, ['UNIQUE_ID', 'DieX', 'DieY'], columns_to_calculate)
    grouped_combined = reorder_columns_and_concatenate(grouped_by_region, grouped_all, columns_to_calculate)

    
    grouped_combined.to_csv(all_output_file_path, index=False)
    shot_m3s.to_csv(shot_output_file_path, index=False)
