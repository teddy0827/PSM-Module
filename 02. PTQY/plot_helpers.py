import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
from config import Config, fontprops

def calculate_lines(center, pitch, max_die, min_die):
    lines = []
    current = center
    for _ in range(max_die + 2):
        lines.append(current)
        current += pitch
    current = center
    for _ in range(abs(min_die) + 1):
        current -= pitch
        lines.append(current)
    return lines

def calculate_statistics(values):
    mean_val = np.mean(values)
    sigma_val = np.std(values)
    m3s_val = np.abs(mean_val) + 3 * sigma_val
    m3s_nm = m3s_val * 1e3  # nm 단위로 변환
    return m3s_nm

def plot_overlay(ax, x, y, dx, dy, v_lines, h_lines,
                 wafer_radius=Config.WAFER_RADIUS,
                 title='Wafer Vector Map',
                 scale_factor=Config.SCALE_FACTOR,
                 show_statistics=True,
                 show_labels=True):
    
    # 벡터 크기 및 색상 맵 계산
    magnitudes = np.sqrt(np.array(dx)**2 + np.array(dy)**2)
    magnitudes_nm = magnitudes * 1e3  # nm 단위로 변환

    norm = plt.Normalize(vmin=magnitudes_nm.min(), vmax=magnitudes_nm.max())
    cmap = plt.cm.jet

    quiver = ax.quiver(x, y, dx, dy, angles='xy', scale_units='xy', scale=scale_factor,
                       color=cmap(norm(magnitudes_nm)), width=0.0015, headwidth=3, headlength=3)

    # 컬러바 추가
    cbar = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax)
    cbar.set_label('Overlay Error Magnitude (nm)')

    # 중앙선 및 다이 라인 표시
    ax.axvline(0, color='red', linewidth=1.0, label='Central X')
    ax.axhline(0, color='red', linewidth=1.0, label='Central Y')

    for vline in v_lines:
        ax.axvline(vline, color='black', linestyle='--', linewidth=0.8)
    for hline in h_lines:
        ax.axhline(hline, color='black', linestyle='--', linewidth=0.8)

    # 웨이퍼 경계
    wafer_circle = Circle((0, 0), wafer_radius, color='green', fill=False,
                          linestyle='-', linewidth=2, label='Wafer Boundary')
    ax.add_patch(wafer_circle)

    # 스케일 바
    scale_bar_label = f'{Config.SCALE_BAR_LENGTH * Config.SCALE_FACTOR * 1e3:.1f}nm'
    scalebar = AnchoredSizeBar(ax.transData,
                               Config.SCALE_BAR_LENGTH_PIXELS,
                               scale_bar_label,
                               Config.SCALE_BAR_POSITION,
                               pad=0.1, color='black',
                               frameon=False,
                               size_vertical=Config.SCALE_BAR_SIZE_VERTICAL,
                               fontproperties=fontprops)
    ax.add_artist(scalebar)

    # 통계 텍스트
    if show_statistics:
        mean_plus_3sigma_x = calculate_statistics(dx)
        mean_plus_3sigma_y = calculate_statistics(dy)
        ax.text(0, Config.TEXT_POSITION_Y,
                f'|m|+3s X: {mean_plus_3sigma_x:.2f} nm\n|m|+3s Y: {mean_plus_3sigma_y:.2f} nm',
                fontsize=10, color='red', ha='center')


    if show_labels:
        ax.set_xlabel('Wafer X Coordinate (wf_x)')
        ax.set_ylabel('Wafer Y Coordinate (wf_y)')
    ax.set_title(title)
    ax.axis('equal')
    ax.grid(False)




    return quiver
