from calculation_module import process_data
from visualization_module import generate_visualization
from wafer_m3s import wafer_m3s_chart

def main():
    # 입력 및 출력 경로 설정
    input_file_path = "data/Delta_PSM.csv"
    shot_output_file_path = "data/output_shot_m3s.csv"
    all_output_file_path = "data/output_all_region_m3s.csv"

    # 데이터 처리
    process_data(input_file_path, shot_output_file_path, all_output_file_path)

    # shot별 m3s 시각화
    generate_visualization(shot_output_file_path)

    # wafer m3s 시각화
    wafer_m3s_chart(all_output_file_path)





if __name__ == "__main__":
    main()

    
