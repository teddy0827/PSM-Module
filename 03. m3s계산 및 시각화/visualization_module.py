import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from datetime import datetime

def generate_visualization(shot_file_path: str):
    # 결과 폴더 생성
    output_folder = "results"
    os.makedirs(output_folder, exist_ok=True)

    # (0) CSV 또는 TSV 읽기
    df = pd.read_csv(shot_file_path)    
    df['P_TIME'] = pd.to_datetime(df['P_TIME'], format='%Y-%m-%d %H:%M')

    # ---------------------------------------------------------
    # (1) 유니크 DieX, DieY 추출
    # ---------------------------------------------------------
    unique_diex = sorted(df['DieX'].unique())
    unique_diey = sorted(df['DieY'].unique())

    n_diex = len(unique_diex)
    n_diey = len(unique_diey)

    # ---------------------------------------------------------
    # (3) P_EQPID & Photo_PPID 조합별로 색상 지정
    # ---------------------------------------------------------
    #  3-1) (P_EQPID, Photo_PPID) 유니크 목록 추출
    unique_pairs = df[['P_EQPID', 'Photo_PPID']].drop_duplicates().reset_index(drop=True)
    n_pairs = len(unique_pairs)

    #  3-2) 컬러맵에서 n_pairs개 만큼 색상 추출 (예: 'tab20' 사용)
    cmap_name = 'Set1'  # 'tab10', 'Set3', 'Paired' 등 원하는 컬러맵을 고를 수 있음
    cmap = plt.get_cmap(cmap_name, n_pairs)

    # ③ color_dict 생성
    color_dict = {}
    for i, row in unique_pairs.iterrows():
        p_eqpid = row['P_EQPID']
        photo_ppid = row['Photo_PPID']
        color_dict[(p_eqpid, photo_ppid)] = cmap(i)

    # ---------------------------------------------------------
    # (4) 시각화 및 저장
    # ---------------------------------------------------------
    columns_to_plot = [
        'delta_psm_x_abs_m3s',
        'delta_psm_y_abs_m3s',
        'ideal_psm_x_abs_m3s',
        'ideal_psm_y_abs_m3s'
    ]

    # 현재 시간 추가
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_pdf = os.path.join(output_folder, f"Shot_M3S_scatter_plots_{current_time}.pdf")

    with PdfPages(output_pdf) as pdf:
        for col in columns_to_plot:
            # 새 Figure 생성
            fig, axes = plt.subplots(
                nrows=n_diey, ncols=n_diex, 
                figsize=(4*n_diex, 3*n_diey),
                sharex=True, sharey=True
            )

            # (DieX, DieY)별 subplot에 scatter
            for dx in unique_diex:
                for dy in unique_diey:
                    row_idx = (n_diey - 1) - unique_diey.index(dy)
                    col_idx = unique_diex.index(dx)
                    ax = axes[row_idx, col_idx]

                    sub_df = df[(df['DieX'] == dx) & (df['DieY'] == dy)]
                    if sub_df.empty:
                        ax.text(0.5, 0.5, 'No Data', ha='center', va='center', transform=ax.transAxes)
                        continue

                    # 카테고리별 색상은 동일하게 사용 가능
                    for (p_eqpid, photo_ppid), group_df in sub_df.groupby(['P_EQPID', 'Photo_PPID']):
                        color_val = color_dict[(p_eqpid, photo_ppid)]
                        ax.scatter(group_df['P_TIME'],
                                group_df[col], 
                                c=[color_val],
                                s=30, alpha=0.8,
                                label=f"{p_eqpid}-{photo_ppid}")

                    ax.set_title(f"DieX={dx}, DieY={dy}")
                    ax.tick_params(axis='x', rotation=45)
                    ax.grid(True, linestyle="--", alpha=0.5)

            # 범례를 맨 마지막에 figure로 추가
            handles, labels = [], []
            for ax_row in axes:
                for ax_sub in (ax_row if isinstance(ax_row, (list, np.ndarray)) else [ax_row]):
                    h, l = ax_sub.get_legend_handles_labels()
                    handles += h
                    labels += l
            hl_dict = dict(zip(labels, handles))  # 유니크 처리

            # 레이아웃 조정 후 범례와 제목 추가
            fig.tight_layout(rect=[0, 0, 1, 0.9])  # 상단 여백 확보
            fig.legend(hl_dict.values(), hl_dict.keys(), loc='upper center', bbox_to_anchor=(0.5, 1.02), ncol=5, prop={'size': 30})  # 범례 폰트 크기 설정
            fig.suptitle(f"Scatter Plot of {col}", y=1.08, fontsize=40)

            # PDF에 Figure 저장 (bbox_inches="tight" 옵션 추가)
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

    print(f"Plots saved to {output_pdf}")
