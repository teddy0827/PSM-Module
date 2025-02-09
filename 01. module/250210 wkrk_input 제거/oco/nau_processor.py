import os
import pandas as pd
import numpy as np  
from datetime import datetime
from openpyxl import load_workbook

def load_excel_sheets(file_path):
    """
    RawData-1 시트를 한 번만 읽어서,
    header가 있는 DataFrame과 header 없이 읽은 DataFrame을 함께 반환합니다.
    """
    rawdata_file = pd.read_excel(file_path, sheet_name='RawData-1')
    rawdata_file_no_header = pd.read_excel(file_path, sheet_name='RawData-1', header=None)
    return rawdata_file, rawdata_file_no_header

def remove_duplicate_files(folder_path):
    """E1 파일만 남기고 E2, E3 파일 삭제"""
    unique_files = {}
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        if os.path.isfile(file_path) and file_name.endswith('.nau'):
            file_name_without_extension = os.path.splitext(file_name)[0]
            split_file_name = file_name_without_extension.split("_")
            base_name = "_".join(split_file_name[:-1])
            file_suffix = split_file_name[-1]
            if base_name not in unique_files:
                if file_suffix == "E1":
                    unique_files[base_name] = file_path
            else:
                if file_suffix in ["E2", "E3"]:
                    try:
                        os.remove(file_path)
                        print(f"중복 파일 {file_name} 삭제 완료")
                    except Exception as e:
                        print(f"{file_name} 파일 삭제 중 에러 발생: {e}")

def extract_file_info(file_name):
    """파일명에서 필요한 정보를 추출하여 반환"""
    file_name_without_extension = os.path.splitext(file_name)[0]
    split_file_name = file_name_without_extension.split("_")
    info_dict = {
        'base_name': "_".join(split_file_name[:-1]),
        'file_suffix': split_file_name[-1],
        'split_file_name': split_file_name
    }
    return info_dict

def process_rawdata_sheet(file_path, info_dict, rawdata_file):
    """RawData-1 시트를 처리 (미리 읽어둔 rawdata_file 사용)"""
    # 추출할 컬럼 위치 설정
    columns_to_extract = [0, 1, 2, 3, 4, 5, 6, 7]
    extracted_data_raw = rawdata_file.iloc[:, columns_to_extract].copy()

    # GROUP 컬럼 추가
    extracted_data_raw['GROUP'] = pd.cut(extracted_data_raw['TEST'], bins=[0, 80, 160, 240], labels=['E1', 'E2', 'E3'])

    # 추가 정보 추출 (여기서는 동일 rawdata_file 사용)
    lot_id_value_raw = rawdata_file.columns[13]
    stepseq_value_raw = rawdata_file.iloc[0, 13]
    wafer_value_raw = rawdata_file.iloc[0, 0]
    p_eqpid_value_raw = rawdata_file.iloc[1, 13]
    photo_ppid_value_raw = rawdata_file.iloc[11, 13]
    p_time_value_raw = rawdata_file.iloc[2, 13]
    m_time_value_raw = rawdata_file.iloc[4, 13]
    chuckid_value_raw = rawdata_file.iloc[15, 13]
    mmo_mrc_eqp_value_raw = rawdata_file.iloc[19, 13]

    # 새로운 컬럼 추가
    extracted_data_raw['STEPSEQ'] = stepseq_value_raw
    extracted_data_raw['LOT_ID'] = lot_id_value_raw

    # 추가 정보 컬럼 추가
    extracted_data_raw['STEP_PITCH_X'] = rawdata_file.iloc[6, 13]
    extracted_data_raw['STEP_PITCH_Y'] = rawdata_file.iloc[7, 13]
    extracted_data_raw['MAP_SHIFT_X'] = rawdata_file.iloc[8, 13]
    extracted_data_raw['MAP_SHIFT_Y'] = rawdata_file.iloc[9, 13]

    # 'coordinate_X', 'coordinate_Y' 매핑
    coord_map = rawdata_file[['Test No', 'coordinate_X', 'coordinate_Y']].drop_duplicates(subset='Test No').set_index('Test No')
    extracted_data_raw['coordinate_X'] = extracted_data_raw['TEST'].map(coord_map['coordinate_X'])
    extracted_data_raw['coordinate_Y'] = extracted_data_raw['TEST'].map(coord_map['coordinate_Y'])

    # 'MRC_RX', 'MRC_RY' 매핑
    coord_map_mrc = rawdata_file[['Test No', 'MRC_RX', 'MRC_RY']].drop_duplicates(subset='Test No').set_index('Test No')
    extracted_data_raw['MRC_RX'] = extracted_data_raw['TEST'].map(coord_map_mrc['MRC_RX'])
    extracted_data_raw['MRC_RY'] = extracted_data_raw['TEST'].map(coord_map_mrc['MRC_RY'])

    # PSM_X, PSM_Y 계산
    mrc_x_minus_mrc_rx = extracted_data_raw['MRC_X'] - extracted_data_raw['MRC_RX']
    mrc_y_minus_mrc_ry = extracted_data_raw['MRC_Y'] - extracted_data_raw['MRC_RY']
    extracted_data_raw['PSM_X'] = 0 - mrc_x_minus_mrc_rx
    extracted_data_raw['PSM_Y'] = 0 - mrc_y_minus_mrc_ry

    # fcp_x, fcp_y 계산
    extracted_data_raw['fcp_x'] = (
        extracted_data_raw['DieX'] * extracted_data_raw['STEP_PITCH_X'] +
        extracted_data_raw['MAP_SHIFT_X']
    )
    extracted_data_raw['fcp_y'] = (
        extracted_data_raw['DieY'] * extracted_data_raw['STEP_PITCH_Y'] +
        extracted_data_raw['MAP_SHIFT_Y']
    )

    # wf_x, wf_y 계산
    extracted_data_raw['wf_x'] = (
        extracted_data_raw['DieX'] * extracted_data_raw['STEP_PITCH_X'] +
        extracted_data_raw['MAP_SHIFT_X'] + extracted_data_raw['coordinate_X']
    )
    extracted_data_raw['wf_y'] = (
        extracted_data_raw['DieY'] * extracted_data_raw['STEP_PITCH_Y'] +
        extracted_data_raw['MAP_SHIFT_Y'] + extracted_data_raw['coordinate_Y']
    )

    # radius 계산
    extracted_data_raw['radius'] = np.sqrt(
        extracted_data_raw['wf_x'] ** 2 + extracted_data_raw['wf_y'] ** 2)

    # 추가 정보 컬럼 추가
    extracted_data_raw['P_EQPID'] = p_eqpid_value_raw
    extracted_data_raw['P_TIME'] = p_time_value_raw
    extracted_data_raw['M_TIME'] = m_time_value_raw
    extracted_data_raw['Photo_PPID'] = photo_ppid_value_raw
    extracted_data_raw['Base_EQP1'] = rawdata_file.iloc[12, 13]
    extracted_data_raw['ChuckID'] = chuckid_value_raw
    extracted_data_raw['ReticleID'] = rawdata_file.iloc[16, 13]
    extracted_data_raw['MMO_MRC_EQP'] = mmo_mrc_eqp_value_raw
    extracted_data_raw['CHIP_X_NUM'] = rawdata_file.iloc[25, 13]
    extracted_data_raw['CHIP_Y_NUM'] = rawdata_file.iloc[26, 13]

    # context 정보추가 (250122)
    extracted_data_raw['Outlier_Spec_Ratio'] = rawdata_file.iloc[20, 13]
    extracted_data_raw['Dmargin_X'] = rawdata_file.iloc[3, 27]
    extracted_data_raw['Dmargin_Y'] = rawdata_file.iloc[4, 27]

    # UNIQUE_ID 컬럼 추가
    extracted_data_raw['UNIQUE_ID'] = extracted_data_raw.apply(
        lambda row: f"{row['STEPSEQ']}_{row['P_EQPID']}_{row['Photo_PPID']}_{row['MMO_MRC_EQP']}_"
                    f"{row['P_TIME']}_{row['M_TIME']}_{row['LOT_ID']}_{row['Wafer']}_{row['GROUP']}", axis=1)
    extracted_data_raw['UNIQUE_ID2'] = extracted_data_raw.apply(
        lambda row: f"{row['STEPSEQ']}_{row['P_EQPID']}_{row['Photo_PPID']}_{row['MMO_MRC_EQP']}_"
                    f"{row['P_TIME']}_{row['M_TIME']}_{row['LOT_ID']}_{row['Wafer']}_{row['TEST']}_"
                    f"{row['DieX']}_{row['DieY']}_{row['GROUP']}", axis=1)
    extracted_data_raw['UNIQUE_ID3'] = extracted_data_raw.apply(
        lambda row: f"{row['STEPSEQ']}_{row['P_EQPID']}_{row['Photo_PPID']}_{row['MMO_MRC_EQP']}_"
                    f"{row['P_TIME']}_{row['M_TIME']}_{row['LOT_ID']}_{row['Wafer']}", axis=1)
    extracted_data_raw['UNIQUE_ID4'] = extracted_data_raw.apply(
        lambda row: f"{row['STEPSEQ']}_{row['P_EQPID']}_{row['Photo_PPID']}_{row['MMO_MRC_EQP']}_"
                    f"{row['P_TIME']}_{row['M_TIME']}_{row['LOT_ID']}_{row['Wafer']}_"
                    f"{row['DieX']}_{row['DieY']}_{row['GROUP']}", axis=1)

    cols_order = [
        'UNIQUE_ID', 'UNIQUE_ID2', 'UNIQUE_ID3', 'UNIQUE_ID4',
        'STEPSEQ', 'LOT_ID', 'Wafer',
        'P_EQPID', 'Photo_PPID', 'P_TIME', 'M_TIME', 'ChuckID', 'ReticleID', 'Base_EQP1', 'MMO_MRC_EQP',
        'GROUP', 'TEST', 'DieX', 'DieY',
        'STEP_PITCH_X', 'STEP_PITCH_Y', 'MAP_SHIFT_X', 'MAP_SHIFT_Y', 'coordinate_X', 'coordinate_Y',
        'fcp_x', 'fcp_y', 'wf_x', 'wf_y', 'radius', 'CHIP_X_NUM', 'CHIP_Y_NUM',
        'Outlier_Spec_Ratio', 'Dmargin_X', 'Dmargin_Y', 
        'X_reg', 'Y_reg', 'MRC_X', 'MRC_Y', 'MRC_RX', 'MRC_RY', 'PSM_X', 'PSM_Y'
    ]
    extracted_data_raw = extracted_data_raw[cols_order]

    return extracted_data_raw

def process_trocs_input_sheet(file_path, info_dict, rawdata_file):
    """Trocs Input 시트를 처리 (추가 정보는 미리 읽은 rawdata_file에서 추출)"""
    trocs_input_file = pd.read_excel(file_path, sheet_name='Trocs Input')

    # 추가 정보 추출
    stepseq_value_raw = rawdata_file.iloc[0, 13]
    lot_id_value_raw = rawdata_file.columns[13]
    wafer_value_raw = rawdata_file.iloc[0, 0]
    p_eqpid_value_raw = rawdata_file.iloc[1, 13]
    photo_ppid_value_raw = rawdata_file.iloc[11, 13]
    p_time_value_raw = rawdata_file.iloc[2, 13]
    m_time_value_raw = rawdata_file.iloc[4, 13]
    chuckid_value_raw = rawdata_file.iloc[15, 13]
    mmo_mrc_eqp_value_raw = rawdata_file.iloc[19, 13]

    trocs_input_file['STEPSEQ'] = stepseq_value_raw
    trocs_input_file['LOT_ID'] = lot_id_value_raw
    trocs_input_file['Wafer'] = wafer_value_raw
    trocs_input_file['P_EQPID'] = p_eqpid_value_raw
    trocs_input_file['Photo_PPID'] = photo_ppid_value_raw
    trocs_input_file['P_TIME'] = p_time_value_raw
    trocs_input_file['M_TIME'] = m_time_value_raw
    trocs_input_file['ChuckID'] = chuckid_value_raw
    trocs_input_file['MMO_MRC_EQP'] = mmo_mrc_eqp_value_raw

    trocs_input_file['UNIQUE_ID'] = trocs_input_file.apply(
        lambda row: f"{row['STEPSEQ']}_{row['P_EQPID']}_{row['Photo_PPID']}_{row['MMO_MRC_EQP']}_"
                    f"{row['P_TIME']}_{row['M_TIME']}_{row['LOT_ID']}_{row['Wafer']}", axis=1)
    trocs_input_file['UNIQUE_ID2'] = trocs_input_file.apply(
        lambda row: f"{row['UNIQUE_ID']}_{row['dCol']}_{row['dRow']}", axis=1)

    cols_to_insert = ['UNIQUE_ID', 'UNIQUE_ID2', 'STEPSEQ', 'LOT_ID', 'Wafer', 'P_EQPID',
                      'Photo_PPID', 'MMO_MRC_EQP', 'P_TIME', 'M_TIME', 'ChuckID']
    for i, col in enumerate(cols_to_insert):
        trocs_input_file.insert(i, col, trocs_input_file.pop(col))

    return trocs_input_file

def process_psm_input_sheet(file_path, info_dict, rawdata_file):
    """PerShotMRC 시트를 처리 (추가 정보는 미리 읽은 rawdata_file에서 추출)"""
    psm_input_file = pd.read_excel(file_path, sheet_name='PerShotMRC')

    # 추가 정보 추출
    stepseq_value_raw = rawdata_file.iloc[0, 13]
    lot_id_value_raw = rawdata_file.columns[13]
    wafer_value_raw = rawdata_file.iloc[0, 0]
    p_eqpid_value_raw = rawdata_file.iloc[1, 13]
    photo_ppid_value_raw = rawdata_file.iloc[11, 13]
    p_time_value_raw = rawdata_file.iloc[2, 13]
    m_time_value_raw = rawdata_file.iloc[4, 13]
    chuckid_value_raw = rawdata_file.iloc[15, 13]
    mmo_mrc_eqp_value_raw = rawdata_file.iloc[19, 13]

    psm_input_file['STEPSEQ'] = stepseq_value_raw
    psm_input_file['LOT_ID'] = lot_id_value_raw
    psm_input_file['Wafer'] = wafer_value_raw
    psm_input_file['P_EQPID'] = p_eqpid_value_raw
    psm_input_file['Photo_PPID'] = photo_ppid_value_raw
    psm_input_file['P_TIME'] = p_time_value_raw
    psm_input_file['M_TIME'] = m_time_value_raw
    psm_input_file['ChuckID'] = chuckid_value_raw
    psm_input_file['MMO_MRC_EQP'] = mmo_mrc_eqp_value_raw

    psm_input_file['UNIQUE_ID'] = psm_input_file.apply(
        lambda row: f"{row['STEPSEQ']}_{row['P_EQPID']}_{row['Photo_PPID']}_{row['MMO_MRC_EQP']}_"
                    f"{row['P_TIME']}_{row['M_TIME']}_{row['LOT_ID']}_{row['Wafer']}", axis=1)
    psm_input_file['UNIQUE_ID2'] = psm_input_file.apply(
        lambda row: f"{row['UNIQUE_ID']}_{row['dCol']}_{row['dRow']}", axis=1)

    cols_to_insert_psm = ['UNIQUE_ID', 'UNIQUE_ID2', 'STEPSEQ', 'LOT_ID', 'Wafer', 'P_EQPID',
                          'Photo_PPID', 'MMO_MRC_EQP', 'P_TIME', 'M_TIME', 'ChuckID']
    for i, col in enumerate(cols_to_insert_psm):
        psm_input_file.insert(i, col, psm_input_file.pop(col))

    return psm_input_file

def process_mrc_data(file_path, info_dict, rawdata_file, rawdata_file_no_header):
    """MRC 데이터를 처리 (추가 정보는 미리 읽은 rawdata_file에서 추출, header-less 데이터는 rawdata_file_no_header 사용)"""
    # 추가 정보 추출
    stepseq_value_raw = rawdata_file.iloc[0, 13]
    lot_id_value_raw = rawdata_file.columns[13]
    wafer_value_raw = rawdata_file.iloc[0, 0]
    p_eqpid_value_raw = rawdata_file.iloc[1, 13]
    photo_ppid_value_raw = rawdata_file.iloc[11, 13]
    p_time_value_raw = rawdata_file.iloc[2, 13]
    m_time_value_raw = rawdata_file.iloc[4, 13]
    chuckid_value_raw = rawdata_file.iloc[15, 13]
    mmo_mrc_eqp_value_raw = rawdata_file.iloc[19, 13]

    mrc_part1 = rawdata_file_no_header.iloc[0:20, 15:17]
    mrc_part2 = rawdata_file_no_header.iloc[22:40, 15:17]
    mrc_part = pd.concat([mrc_part1, mrc_part2], ignore_index=True)
   

    mrc_part.columns = ['K PARA', 'GPM']
    mrc_part['INDEX'] = range(1, len(mrc_part) + 1)

    mrc_part['STEPSEQ'] = stepseq_value_raw
    mrc_part['LOT_ID'] = lot_id_value_raw
    mrc_part['Wafer'] = wafer_value_raw
    mrc_part['P_EQPID'] = p_eqpid_value_raw
    mrc_part['Photo_PPID'] = photo_ppid_value_raw
    mrc_part['P_TIME'] = p_time_value_raw
    mrc_part['M_TIME'] = m_time_value_raw
    mrc_part['ChuckID'] = chuckid_value_raw
    mrc_part['MMO_MRC_EQP'] = mmo_mrc_eqp_value_raw

    mrc_part['UNIQUE_ID'] = mrc_part.apply(
        lambda row: f"{row['STEPSEQ']}_{row['P_EQPID']}_{row['Photo_PPID']}_{row['MMO_MRC_EQP']}_"
                    f"{row['P_TIME']}_{row['M_TIME']}_{row['LOT_ID']}_{row['Wafer']}", axis=1)

    mrc_cols_order = [
        'UNIQUE_ID',
        'STEPSEQ', 'LOT_ID', 'Wafer',
        'P_EQPID', 'Photo_PPID', 'MMO_MRC_EQP', 'P_TIME', 'M_TIME', 'ChuckID',
        'K PARA', 'GPM', 'INDEX'
    ]
    mrc_part = mrc_part[mrc_cols_order]

    return mrc_part


def process_wk_rk_input_data(file_path, info_dict, rawdata_file):
    """
    RawData-1 시트에서 APC Input (WK, RK)값 및 관련 좌표를 추출하여 DataFrame으로 반환합니다.
    (이 예제에서는 'APC_WK'와 'APC_RK'라는 컬럼명이 존재한다고 가정합니다.)
    """
     
    # 추가 정보 추출
    stepseq_value_raw = rawdata_file.iloc[0, 13]
    lot_id_value_raw = rawdata_file.columns[13]
    wafer_value_raw = rawdata_file.iloc[0, 0]
    p_eqpid_value_raw = rawdata_file.iloc[1, 13]
    photo_ppid_value_raw = rawdata_file.iloc[11, 13]
    p_time_value_raw = rawdata_file.iloc[2, 13]
    m_time_value_raw = rawdata_file.iloc[4, 13]
    chuckid_value_raw = rawdata_file.iloc[15, 13]
    mmo_mrc_eqp_value_raw = rawdata_file.iloc[19, 13]

    # APC Input 데이터 추출
    wk_input = rawdata_file.iloc[0:20, 96]  # (20,)
    rk_input = rawdata_file.iloc[2:20, 97]  # (18,) RK1,2는 제외

    # 두 Series를 세로로 연결하여 (38,1) 형태의 DataFrame 생성
    wk_rk_input = pd.concat([wk_input, rk_input], axis=0, ignore_index=True).to_frame()
    
    # 'GPM' 컬럼명으로 변경
    wk_rk_input.columns = ['GPM']

    # 'K PARA' 컬럼 추가 (W1~W20, R3~R20)
    k_para_labels = [f"W{i+1}" for i in range(20)] + [f"R{i+3}" for i in range(18)]
    wk_rk_input['K PARA'] = k_para_labels
    

    wk_rk_input['INDEX'] = range(1, len(wk_rk_input) + 1)
    wk_rk_input['STEPSEQ'] = stepseq_value_raw
    wk_rk_input['LOT_ID'] = lot_id_value_raw
    wk_rk_input['Wafer'] = wafer_value_raw
    wk_rk_input['P_EQPID'] = p_eqpid_value_raw
    wk_rk_input['Photo_PPID'] = photo_ppid_value_raw
    wk_rk_input['P_TIME'] = p_time_value_raw
    wk_rk_input['M_TIME'] = m_time_value_raw
    wk_rk_input['ChuckID'] = chuckid_value_raw
    wk_rk_input['MMO_MRC_EQP'] = mmo_mrc_eqp_value_raw

    wk_rk_input['UNIQUE_ID'] = wk_rk_input.apply(
        lambda row: f"{row['STEPSEQ']}_{row['P_EQPID']}_{row['Photo_PPID']}_{row['MMO_MRC_EQP']}_"
                    f"{row['P_TIME']}_{row['M_TIME']}_{row['LOT_ID']}_{row['Wafer']}", axis=1)

    wk_rk_cols_order = [
        'UNIQUE_ID',
        'STEPSEQ', 'LOT_ID', 'Wafer',
        'P_EQPID', 'Photo_PPID', 'MMO_MRC_EQP', 'P_TIME', 'M_TIME', 'ChuckID',
        'K PARA', 'GPM', 'INDEX'
    ]
    wk_rk_input = wk_rk_input[wk_rk_cols_order]

    return wk_rk_input



def process_nau_file(file_path):
    """하나의 nau 파일을 처리하여 데이터프레임 반환"""
    file_name = os.path.basename(file_path)
    info_dict = extract_file_info(file_name)

    # RawData-1 시트를 한 번만 읽기 (header 포함 및 header 없이)
    rawdata_file, rawdata_file_no_header = load_excel_sheets(file_path)

    rawdata_df = process_rawdata_sheet(file_path, info_dict, rawdata_file)
    trocs_input_df = process_trocs_input_sheet(file_path, info_dict, rawdata_file)
    psm_input_df = process_psm_input_sheet(file_path, info_dict, rawdata_file)
    mrc_df = process_mrc_data(file_path, info_dict, rawdata_file, rawdata_file_no_header)
    wk_rk_input_df = process_wk_rk_input_data(file_path, info_dict, rawdata_file)

    return rawdata_df, trocs_input_df, psm_input_df, mrc_df, wk_rk_input_df

def save_combined_data(rawdata_list, trocs_input_list, psm_input_list, mrc_list, wk_rk_input_list):
    """데이터프레임 리스트를 병합하여 파일로 저장"""
    combined_rawdata = pd.concat(rawdata_list, ignore_index=True)
    combined_trocs_input = pd.concat(trocs_input_list, ignore_index=True)
    combined_psm_input = pd.concat(psm_input_list, ignore_index=True)
    combined_mrc_data = pd.concat(mrc_list, ignore_index=True)
    combined_wk_rk_input = pd.concat(wk_rk_input_list, ignore_index=True)

    combined_rawdata = combined_rawdata.sort_values(by=['UNIQUE_ID', 'TEST', 'DieX', 'DieY'])
    combined_trocs_input = combined_trocs_input.sort_values(by=['UNIQUE_ID', 'dCol', 'dRow'])
    combined_psm_input = combined_psm_input.sort_values(by=['UNIQUE_ID', 'dCol', 'dRow'])
    combined_mrc_data = combined_mrc_data.sort_values(by=['UNIQUE_ID', 'INDEX'])
    combined_wk_rk_input = combined_wk_rk_input.sort_values(by=['UNIQUE_ID', 'INDEX'])

    combined_rawdata.to_csv('RawData-1.csv', index=False)
    combined_trocs_input.to_csv('Trocs_Input.csv', index=False)
    combined_psm_input.to_csv('PerShotMRC.csv', index=False)
    combined_mrc_data.to_csv('MRC.csv', index=False)
    combined_wk_rk_input.to_csv('WK_RK_INPUT.csv', index=False)

def main():
    folder_path = 'C:/py_data/nau/2lot'
    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} 작업 시작")

    remove_duplicate_files(folder_path)
    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} 중복 파일 제거 완료")

    rawdata_list = []
    trocs_input_list = []
    psm_input_list = []
    mrc_list = []
    wk_rk_input_list = []

    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} nau 파일 처리 시작")
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.nau'):
            file_path = os.path.join(folder_path, file_name)
            try:
                rawdata_df, trocs_input_df, psm_input_df, mrc_df, wk_rk_input_df = process_nau_file(file_path)
                rawdata_list.append(rawdata_df)
                trocs_input_list.append(trocs_input_df)
                psm_input_list.append(psm_input_df)
                mrc_list.append(mrc_df)
                wk_rk_input_list.append(wk_rk_input_df)
                print(f"{file_name} 처리 완료")
            except Exception as e:
                print(f"{file_name} 처리 중 에러 발생: {e}")
    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} nau 파일 처리 완료")

    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} 데이터 저장 시작")

    save_combined_data(rawdata_list, trocs_input_list, psm_input_list, mrc_list, wk_rk_input_list)
    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} 데이터 저장 완료")

    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} 작업 완료")

if __name__ == "__main__":
    main()



