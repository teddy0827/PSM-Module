import pandas as pd
from datetime import datetime
import os
import logging

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("process_log.log"),
        logging.StreamHandler()
    ]
)


from nau_processor import (
    remove_duplicate_files,
    process_nau_file,
    save_combined_data
)

from calc_regression import (
    kmrc_decorrect,
    remove_psm_add_pointmrc,
    multi_lot_regression_and_fitting,
    reorder_coefficients,
    psm_decorrect,
    resi_to_cpe,
    cpe19p_fitting,
    ideal_psm,
    delta_psm
)

from zernike_analysis import zernike_analysis


def main():
    folder_path = 'C:/py_data/nau/2lot'
    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} 작업 시작")

    remove_duplicate_files(folder_path)
    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} 중복 파일 제거 완료")


    ############################### 1. nau_processor ########################################

    rawdata_list = []
    trocs_input_list = []
    psm_input_list = []
    mrc_list = []

    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} nau 파일 처리 시작")
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.nau'):
            file_path = os.path.join(folder_path, file_name)
            try:
                rawdata_df, trocs_input_df, psm_input_df, mrc_df = process_nau_file(file_path)
                rawdata_list.append(rawdata_df)
                trocs_input_list.append(trocs_input_df)
                psm_input_list.append(psm_input_df)
                mrc_list.append(mrc_df)
                print(f"{file_name} 처리 완료")
            except Exception as e:
                print(f"{file_name} 처리 중 에러 발생: {e}")
    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} nau 파일 처리 완료")

    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} 데이터 저장 시작")
    save_combined_data(rawdata_list, trocs_input_list, psm_input_list, mrc_list)
    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} 데이터 저장 완료")

    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} 작업 완료")


    ############################### 2. calc_regression ########################################


    # 데이터 로드
    df_rawdata = pd.read_csv("RawData-1.csv")
    df_mrc_input = pd.read_csv("MRC.csv")

    # MRC Decorrect
    df_mrc_de = kmrc_decorrect(df_rawdata, df_mrc_input)
    df_rawdata = pd.concat([df_rawdata, df_mrc_de], axis=1)
    print(datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 'MRC Decorrect 완료.')

    # Raw calculation
    df_rawdata = remove_psm_add_pointmrc(df_rawdata)
    print(datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 'Raw 처리 완료.')

    # 회귀 분석 및 예측 수행
    df_coeff = multi_lot_regression_and_fitting(df_rawdata)

    # 회귀 계수의 컬럼 순서 재정렬
    df_coeff = reorder_coefficients(df_coeff)

    # 회귀 계수 저장
    df_coeff.to_csv('OSR_K.csv', index=False)
    print(datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 'OSR Regression 및 Fitting 완료.')

    # PSM Decorrect
    df_psm_input = pd.read_csv('PerShotMRC.csv')
    df_psm_de = psm_decorrect(df_rawdata, df_psm_input)
    df_rawdata = pd.concat([df_rawdata, df_psm_de], axis=1)
    print(datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 'PSM Decorrect 완료.')

    # CPE 19-parameter modeling
    df_cpe19p = resi_to_cpe(df_rawdata)
    df_cpe19p.to_csv('CPE_19p.csv', index=False)
    print(datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 'CPE 19para Regression 완료.')

    # CPE fitting
    df_cpe19p_fit_res = cpe19p_fitting(df_rawdata, df_cpe19p)
    df_rawdata = pd.concat([df_rawdata, df_cpe19p_fit_res], axis=1)
    print(datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 'CPE 19para Fitting 완료.')

    # Ideal PSM
    df_rawdata = ideal_psm(df_rawdata)
    print(datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 'Ideal PSM 완료.')

    # Delta PSM
    df_rawdata = delta_psm(df_rawdata)
    df_rawdata.to_csv('Delta_PSM.csv', index=False)
    print(datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 'Delta PSM 완료.')
    print(datetime.now().strftime('%Y-%m-%d %H:%M:%S'), '모든 작업이 완료되었습니다.')

    ############################### 3. zernike_analysis ########################################
    
    # Zernike 분석 실행
    max_index = 64
    logging.info("Starting Zernike analysis")
    df_z_coeff, df_rawdata = zernike_analysis(df_rawdata, max_index=max_index)

    # Z계수 nm단위로 결과 저장
    df_z_coeff.to_csv("Fringe_Zernike_Coefficients.csv", index=False)
    logging.info("Zernike coefficients saved to Zernike_Coefficients.csv")

    df_rawdata.to_csv("Zernike_Fit.csv", index=False)
    logging.info("Zernike predictions and residuals saved to Z_FIT.csv")




if __name__ == "__main__":
    main()
    
