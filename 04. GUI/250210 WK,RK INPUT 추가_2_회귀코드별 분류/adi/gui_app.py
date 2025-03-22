
# GUI 애플리케이션 클래스
# - GUI를 통해 사용자 입력을 받고, 처리 과정을 실행하는 클래스
# - GUIApp 클래스는 tkinter의 Tk 클래스를 상속받아 구현되었습니다.  
# - GUIApp 클래스의 생성자에서 GUI 위젯들을 생성하고 배치하며, 각 위젯의 이벤트 처리 메서드를 구현합니다.
# - browse_nau_folder 메서드에서는 폴더 선택 대화상자를 통해 Nau 폴더를 선택하고, 선택한 폴더 경로를 입력 위젯에 표시합니다.
# - start_process 메서드에서 처리 과정을 실행하는 스레드를 생성하고 시작합니다.
# - run_workflow 메서드에서 실제 처리 과정을 수행합니다.
# - 입력값을 읽고 검증하는 부분에서는 입력값을 읽은 후, 각 값이 유효한지 확인하기 위해 try-except 구문을 사용합니다.   
# - 처리 과정 실행 중에 발생하는 예외를 처리하기 위해 try-except 구문을 사용합니다.
# - 로그 출력용 스크롤 텍스트 위젯을 사용하기 위해 tkinter의 scrolledtext 모듈을 import합니다.
# - 로그 출력용 스크롤 텍스트 위젯은 읽기 전용 상태로 설정되어 있습니다.
# - 로그 출력용 스크롤 텍스트 위젯에 메시지를 출력하는 log 메서드를 구현합니다.

import os
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext
from datetime import datetime

# 기존 모듈 import (각 파일은 기존 코드대로 작성되어 있다고 가정)
from nau_processor import remove_duplicate_files, process_nau_file, save_combined_data
from raduis_filter import filter_by_radius
from calc_regression_adi import (
    kmrc_decorrect,
    wk_rk_input_decorrect,
    remove_psm_add_pointmrc,
    multi_lot_regression_and_fitting,
    detect_outliers_studentized_residual,
    reorder_coefficients,
    psm_decorrect,
    resi_to_cpe,
    cpe19p_fitting,
    ideal_psm,
    delta_psm
)
from zernike_analysis_adi import zernike_analysis

import pandas as pd

from config import FOLDER_PATH, RADIUS_THRESHOLD, OUTLIER_THRESHOLD, DMARGIN_X, DMARGIN_Y, OUTLIER_SPEC_RATIO, MAX_INDEX
# design_matrix_config에서 기본 옵션을 가져옵니다.
from design_matrix_config import DEFAULT_KMRC_OPTION, DEFAULT_PSM_OPTION, DEFAULT_CPE_OPTION

class GUIApp:
    def __init__(self, master):
        self.master = master
        master.title("Overlay Regression & Analysis GUI")
        
        # Nau 폴더 선택: 기본값을 config에서 가져옴
        tk.Label(master, text="Nau Folder:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        self.entry_nau_folder = tk.Entry(master, width=50)
        self.entry_nau_folder.grid(row=0, column=1, padx=5, pady=5)
        self.entry_nau_folder.insert(0, FOLDER_PATH)
        tk.Button(master, text="Browse", command=self.browse_nau_folder).grid(row=0, column=2, padx=5, pady=5)
        
        # Radius Threshold: config 값 사용
        tk.Label(master, text="Radius Threshold:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        self.entry_radius = tk.Entry(master, width=20)
        self.entry_radius.insert(0, str(RADIUS_THRESHOLD))
        self.entry_radius.grid(row=1, column=1, sticky=tk.W, padx=5, pady=5)
        
        # Outlier Threshold: config 값 사용
        tk.Label(master, text="Outlier Threshold:").grid(row=2, column=0, sticky=tk.W, padx=5, pady=5)
        self.entry_outlier = tk.Entry(master, width=20)
        self.entry_outlier.insert(0, str(OUTLIER_THRESHOLD))
        self.entry_outlier.grid(row=2, column=1, sticky=tk.W, padx=5, pady=5)
        
        # Dmargin X: config 값 사용
        tk.Label(master, text="Dmargin X:").grid(row=3, column=0, sticky=tk.W, padx=5, pady=5)
        self.entry_dmargin_x = tk.Entry(master, width=20)
        self.entry_dmargin_x.insert(0, str(DMARGIN_X))
        self.entry_dmargin_x.grid(row=3, column=1, sticky=tk.W, padx=5, pady=5)
        
        # Dmargin Y: config 값 사용
        tk.Label(master, text="Dmargin Y:").grid(row=4, column=0, sticky=tk.W, padx=5, pady=5)
        self.entry_dmargin_y = tk.Entry(master, width=20)
        self.entry_dmargin_y.insert(0, str(DMARGIN_Y))
        self.entry_dmargin_y.grid(row=4, column=1, sticky=tk.W, padx=5, pady=5)
        
        # Outlier Spec Ratio: config 값 사용
        tk.Label(master, text="Outlier Spec Ratio:").grid(row=5, column=0, sticky=tk.W, padx=5, pady=5)
        self.entry_outlier_ratio = tk.Entry(master, width=20)
        self.entry_outlier_ratio.insert(0, str(OUTLIER_SPEC_RATIO))
        self.entry_outlier_ratio.grid(row=5, column=1, sticky=tk.W, padx=5, pady=5)
        
        # Max Zernike Index: config 값 사용
        tk.Label(master, text="Max Zernike Index:").grid(row=6, column=0, sticky=tk.W, padx=5, pady=5)
        self.entry_zernike = tk.Entry(master, width=20)
        self.entry_zernike.insert(0, str(MAX_INDEX))
        self.entry_zernike.grid(row=6, column=1, sticky=tk.W, padx=5, pady=5)
        
        # ---------------- 디자인 행렬 옵션 선택 ----------------
        # KMRC Design Matrix Option
        tk.Label(master, text="KMRC Option:").grid(row=7, column=0, sticky=tk.W, padx=5, pady=5)
        self.kmrc_option = tk.StringVar()
        self.kmrc_option.set(DEFAULT_KMRC_OPTION)  # 예: "15para"
        self.optionmenu_kmrc = tk.OptionMenu(master, self.kmrc_option, "19para", "18para")
        self.optionmenu_kmrc.grid(row=7, column=1, sticky=tk.W, padx=5, pady=5)
        
        # PSM Design Matrix Option
        tk.Label(master, text="PSM Option:").grid(row=8, column=0, sticky=tk.W, padx=5, pady=5)
        self.psm_option = tk.StringVar()
        self.psm_option.set(DEFAULT_PSM_OPTION)  # 예: "38para"
        self.optionmenu_psm = tk.OptionMenu(master, self.psm_option, "33para", "38para")
        self.optionmenu_psm.grid(row=8, column=1, sticky=tk.W, padx=5, pady=5)
        
        # CPE Design Matrix Option
        tk.Label(master, text="CPE Option:").grid(row=9, column=0, sticky=tk.W, padx=5, pady=5)
        self.cpe_option = tk.StringVar()
        self.cpe_option.set(DEFAULT_CPE_OPTION)  # 예: "15para"
        self.optionmenu_cpe = tk.OptionMenu(master, self.cpe_option, "19para", "6para", "15para")
        self.optionmenu_cpe.grid(row=9, column=1, sticky=tk.W, padx=5, pady=5)
        # -------------------------------------------------------
        
        # 실행 버튼
        self.button_run = tk.Button(master, text="Run Process", command=self.start_process)
        self.button_run.grid(row=10, column=1, padx=5, pady=10)
        
        # 로그 출력용 스크롤 텍스트 위젯
        self.log_text = scrolledtext.ScrolledText(master, width=80, height=20, state='disabled')
        self.log_text.grid(row=11, column=0, columnspan=3, padx=5, pady=5)
    
    def log(self, message):
        # 현재 시간을 타임스탬프로 추가하여 로그 메시지 출력
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        full_message = f"[{timestamp}] {message}"
        self.log_text.config(state='normal')
        self.log_text.insert(tk.END, full_message + "\n")
        self.log_text.see(tk.END)
        self.log_text.config(state='disabled')
    
    def browse_nau_folder(self):
        folder = filedialog.askdirectory()
        if folder:
            self.entry_nau_folder.delete(0, tk.END)
            self.entry_nau_folder.insert(0, folder)
    
    def start_process(self):
        # 입력값 읽기 및 검증
        nau_folder = self.entry_nau_folder.get().strip()
        if not nau_folder or not os.path.isdir(nau_folder):
            messagebox.showerror("Error", "Please select a valid Nau folder.")
            return
        try:
            radius_threshold = float(self.entry_radius.get().strip())
            outlier_threshold = float(self.entry_outlier.get().strip())
            dmargin_x = float(self.entry_dmargin_x.get().strip())
            dmargin_y = float(self.entry_dmargin_y.get().strip())
            outlier_ratio = float(self.entry_outlier_ratio.get().strip())
            max_zernike_index = int(self.entry_zernike.get().strip())
        except ValueError:
            messagebox.showerror("Error", "Please enter valid numeric parameters.")
            return
        
        self.button_run.config(state='disabled')
        threading.Thread(
            target=self.run_workflow,
            args=(nau_folder, radius_threshold, outlier_threshold, dmargin_x, dmargin_y, outlier_ratio, max_zernike_index),
            daemon=True
        ).start()
    
    def run_workflow(self, nau_folder, radius_threshold, outlier_threshold, dmargin_x, dmargin_y, outlier_ratio, max_zernike_index):
        try:
            self.log("Starting process...")
            
            # 결과 저장 폴더 지정
            results_dir = "results"
            if not os.path.exists(results_dir):
                os.makedirs(results_dir)
            
            # 1. Nau 파일 처리
            self.log("Removing duplicate files...")
            remove_duplicate_files(nau_folder)
            self.log("Duplicate files removed.")
            
            rawdata_list = []
            trocs_input_list = []
            psm_input_list = []
            mrc_list = []
            wk_rk_input_list = []
            
            self.log("Processing .nau files...")
            nau_files = [f for f in os.listdir(nau_folder) if f.endswith('.nau')]
            if not nau_files:
                self.log("No .nau files found in the selected folder.")
                self.button_run.config(state='normal')
                return
            
            for file_name in nau_files:
                file_path = os.path.join(nau_folder, file_name)
                self.log(f"Processing file: {file_name} ...")
                try:
                    rawdata_df, trocs_input_df, psm_input_df, mrc_df, wk_rk_input_df = process_nau_file(file_path)
                    rawdata_list.append(rawdata_df)
                    trocs_input_list.append(trocs_input_df)
                    psm_input_list.append(psm_input_df)
                    mrc_list.append(mrc_df)
                    wk_rk_input_list.append(wk_rk_input_df)
                    self.log(f"File {file_name} processed successfully.")
                except Exception as e:
                    self.log(f"Error processing {file_name}: {e}")
            
            self.log("Saving combined data as 'RawData-1.csv' ...")
            save_combined_data(rawdata_list, trocs_input_list, psm_input_list, mrc_list, wk_rk_input_list)
            self.log("Combined data saved in results folder.")
            
            # 2. Radius Filtering
            self.log("Starting radius filtering...")
            input_path = os.path.join(results_dir, "RawData-1.csv")
            filtered_data_path = os.path.join(results_dir, "Radius_Filtered_RawData-1.csv")
            filter_by_radius(input_path, filtered_data_path, radius_threshold)
            self.log("Radius filtering completed.")
            
            # 3. Regression & Residual Analysis
            self.log("Loading filtered data and regression inputs...")
            df_rawdata = pd.read_csv(filtered_data_path)
            df_mrc_input = pd.read_csv(os.path.join(results_dir, "MRC.csv"))
            df_wk_rk_input = pd.read_csv(os.path.join(results_dir, "WK_RK_INPUT.csv"))
            
            self.log("Performing MRC decorrection...")
            df_mrc_de = kmrc_decorrect(df_rawdata, df_mrc_input)
            df_rawdata = pd.concat([df_rawdata, df_mrc_de], axis=1)
            self.log("MRC decorrection completed.")
            
            self.log("Performing WK/RK Input decorrection...")
            df_mrcwkrk_de = wk_rk_input_decorrect(df_rawdata, df_wk_rk_input)
            df_rawdata = pd.concat([df_rawdata, df_mrcwkrk_de], axis=1)
            self.log("WK/RK Input decorrection completed.")
            
            self.log("Processing raw data for regression...")
            df_rawdata = remove_psm_add_pointmrc(df_rawdata)
            self.log("Raw data processed.")
            
            self.log("Performing multi-lot regression and fitting...")
            df_coeff = multi_lot_regression_and_fitting(df_rawdata, kmrc_option=self.kmrc_option.get())
            
            self.log("Detecting outliers...")
            df_rawdata = detect_outliers_studentized_residual(
                df_rawdata, threshold=outlier_threshold, group_col='UNIQUE_ID',
                dmargin_x=dmargin_x, dmargin_y=dmargin_y, outlier_spec_ratio=outlier_ratio
            )
            
            self.log("Reordering regression coefficients...")
            df_coeff = reorder_coefficients(df_coeff)
            df_coeff.to_csv(os.path.join(results_dir, 'OSR_K.csv'), index=False)
            self.log("Regression coefficients saved as 'OSR_K.csv' in results folder.")
            
            self.log("Performing PSM decorrection...")
            df_psm_input = pd.read_csv(os.path.join(results_dir, 'PerShotMRC.csv'))
            df_psm_de = psm_decorrect(df_rawdata, df_psm_input, psm_option=self.psm_option.get())
            df_rawdata = pd.concat([df_rawdata, df_psm_de], axis=1)
            self.log("PSM decorrection completed.")
            
            self.log("Performing CPE 19-parameter modeling...")
            df_cpe19p = resi_to_cpe(df_rawdata, cpe_option=self.cpe_option.get())
            df_cpe19p.to_csv(os.path.join(results_dir, 'CPE_19p.csv'), index=False)
            self.log("CPE 19-parameter regression completed and saved as 'CPE_19p.csv' in results folder.")
            
            self.log("Performing CPE 19-parameter fitting...")
            df_cpe19p_fit_res = cpe19p_fitting(df_rawdata, df_cpe19p)
            df_rawdata = pd.concat([df_rawdata, df_cpe19p_fit_res], axis=1)
            self.log("CPE 19-parameter fitting completed.")
            
            self.log("Performing Ideal PSM correction...")
            df_rawdata = ideal_psm(df_rawdata)
            self.log("Ideal PSM completed.")
            
            self.log("Calculating Delta PSM...")
            df_rawdata = delta_psm(df_rawdata)
            df_rawdata.to_csv(os.path.join(results_dir, 'Delta_PSM.csv'), index=False)
            self.log("Delta PSM completed and saved as 'Delta_PSM.csv' in results folder.")
            
            # 4. Zernike Analysis
            self.log("Performing Zernike analysis...")
            df_z_coeff, df_rawdata = zernike_analysis(df_rawdata, max_index=max_zernike_index)
            df_z_coeff.to_csv(os.path.join(results_dir, "Fringe_Zernike_Coefficients.csv"), index=False)
            self.log("Zernike analysis completed and coefficients saved as 'Fringe_Zernike_Coefficients.csv' in results folder.")
            
            self.log("All processes completed successfully.")
        except Exception as e:
            self.log(f"An error occurred: {e}")
        finally:
            self.button_run.config(state='normal')

if __name__ == "__main__":
    root = tk.Tk()
    app = GUIApp(root)
    root.mainloop()

