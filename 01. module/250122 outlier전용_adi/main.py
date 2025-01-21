''' 
※ 돌리기전 주의사항
1. nau 저장폴더 설정.  기본 : 'C:/py_data/nau/2lot'
2. Edge Clear 위해서 radius_threshold 설정해여함. 기본 150000 설정되어 있음.
3. Zernike 분석시 max_index 설정해야함. 기본 64 설정되어 있음.

'''


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

from calc_regression_adi import (
    kmrc_decorrect,
    remove_psm_add_pointmrc,
    multi_lot_regression_and_fitting,
    detect_outliers_studentized_residual,  # outlier 판정용 추가 
    reorder_coefficients

)


from raduis_filter import filter_by_radius  



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


    ############################### 2. raduis_filter ######################################## 

    # RawData 파일 필터링
    input_path = "RawData-1.csv"
    filtered_data_path = "Radius_Filtered_RawData-1.csv"
    radius_threshold = 150000  # 원하는 radius 임계값 설정

    print("Starting radius filtering...")
    filter_by_radius(input_path, filtered_data_path, radius_threshold)


    ############################### 3. calc_regression ########################################


    # 데이터 로드
    df_rawdata = pd.read_csv("Radius_Filtered_RawData-1.csv")
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


    ################################### outlier 판정용 추가 ######################################################################
    # outlier 판정
    df_coeff = detect_outliers_studentized_residual(df_rawdata, threshold=2.0, group_col='UNIQUE_ID',
                                                    dmargin_x=0.005, dmargin_y=0.0025, outlier_spec_ratio=0.5)

    
    ########################################################################################################

    df_rawdata.to_csv('outlier.csv', index=False)



    # 회귀 계수의 컬럼 순서 재정렬
    df_coeff = reorder_coefficients(df_coeff)

    # 회귀 계수 저장
    df_coeff.to_csv('OSR_K.csv', index=False)
    print(datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 'OSR Regression 및 Fitting 완료.')



if __name__ == "__main__":
    main()
    
