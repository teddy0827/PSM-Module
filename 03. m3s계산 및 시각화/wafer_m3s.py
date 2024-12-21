import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from datetime import datetime
import os

def wafer_m3s_chart(all_output_file_path: str):        
    # 데이터 준비
    excel_data = pd.read_csv(all_output_file_path) 

    # "-"와 같은 잘못된 시간 문자열을 NaN으로 대체
    excel_data['P_TIME'] = excel_data['P_TIME'].replace("-", pd.NaT)

    # 'P_TIME'을 시간 형식으로 변환
    excel_data['P_TIME'] = pd.to_datetime(excel_data['P_TIME'], errors='coerce')

    # 세 가지 컬럼을 합쳐서 고유한 그룹 생성
    excel_data['group'] = excel_data[['P_EQPID', 'Photo_PPID', 'MMO_MRC_EQP', 'GROUP']].agg('_'.join, axis=1)  # M_STEP 삭제처리함

    # 고유 그룹별 색상을 사용할 수 있도록 정의
    unique_groups = excel_data['group'].unique()

    # 선명한 색상 팔레트에서 고유 그룹에 색상 할당
    colors_list = list(mcolors.TABLEAU_COLORS.values())
    group_color_map = {group: colors_list[idx % len(colors_list)] for idx, group in enumerate(unique_groups)}

    # 새로운 y축 열 목록
    new_y_columns = [
        'ideal_psm_x_m3s', 'ideal_psm_y_m3s', 'delta_psm_x_m3s', 'delta_psm_y_m3s'
    ]

    # 데이터프레임에서 고유한 지역(region) 목록 가져오기
    regions = excel_data['region'].unique()  # 'region' 컬럼 이름이 다르다면 수정 필요

    # 현재 시간 가져오기
    current_time = datetime.now().strftime('%Y%m%d_%H%M%S')

    # 결과 저장 경로
    save_dir = 'results'
    os.makedirs(save_dir, exist_ok=True)  # 폴더가 없으면 생성

    # 각 지역별로 그래프 그리기
    for region in regions:
        region_data = excel_data[excel_data['region'] == region]  # 해당 지역 데이터 필터링

        plt.figure(figsize=(20, 25))
        plt.suptitle(f'Region: {region}', fontsize=16)  # 지역명 제목 추가
        
        for idx, y_column in enumerate(new_y_columns, 1):
            plt.subplot(6, 4, idx)
            
            # 그룹에 따른 색상 구분
            for group in region_data['group'].unique():
                group_data = region_data[region_data['group'] == group]
                plt.scatter(group_data['P_TIME'], group_data[y_column], label=group, color=group_color_map[group])
            
            plt.title(f'{y_column} vs P_TIME')
            plt.xlabel('P_TIME')
            plt.ylabel(y_column)
            plt.grid(True)
            plt.xticks(rotation=45)
            plt.tight_layout()

        # 범례를 첫 번째 행의 가장 오른쪽에 위치
        plt.subplot(6, 4, 4)
        plt.legend(title='Group (P_EQPID, Photo_PPID, MMO_MRC_EQP, GROUP)', bbox_to_anchor=(1.25, 1), loc='upper left')  # M_STEP 삭제처리함

        # 그래프 저장
        save_path = f"{save_dir}/Region_{region}_{current_time}.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot for region {region} at: {save_path}")
        
        plt.close()  # 메모리 해제
