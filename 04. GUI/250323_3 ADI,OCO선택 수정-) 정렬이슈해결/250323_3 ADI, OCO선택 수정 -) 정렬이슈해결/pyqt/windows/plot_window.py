import matplotlib.pyplot as plt
import numpy as np
from PyQt5.QtWidgets import QMainWindow
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from plot_helpers import plot_overlay, calculate_lines
from windows.enlarged_plot_window import EnlargedPlotWindow
from config import Config



class PlotWindow(QMainWindow):
    def __init__(self, unique_id, df):
        super().__init__()
        self.setWindowTitle(f"Plot for {unique_id}")
        self.setGeometry(100, 100, 1200, 800)

        df_lot = df[df['UNIQUE_ID'] == unique_id].copy()
        df_lot['radius'] = np.sqrt(df_lot['wf_x']**2 + df_lot['wf_y']**2)

        conditions = [
            (df_lot['radius'] <= 50000),
            (df_lot['radius'] > 50000) & (df_lot['radius'] <= 100000),
            (df_lot['radius'] > 100000) & (df_lot['radius'] <= 150000)
        ]
        choices = ['Center', 'Middle', 'Edge']
        df_lot['region'] = np.select(conditions, choices, default='Outside')
        self.df_lot = df_lot

        self.enlarged_plot_windows = []
        self.cid = None

        self.init_ui()

    def init_ui(self):
        from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox
        main_widget = QWidget()
        main_layout = QVBoxLayout()

        region_label = QLabel("Select Region:")
        self.region_combo = QComboBox()
        self.region_combo.addItems(['All', 'Center', 'Middle', 'Edge'])
        self.region_combo.setCurrentIndex(0)
        self.region_combo.currentIndexChanged.connect(self.update_plot)

        region_layout = QHBoxLayout()
        region_layout.addWidget(region_label)
        region_layout.addWidget(self.region_combo)

        self.fig, self.axes = plt.subplots(3, 5, figsize=(12, 8))
        self.canvas = FigureCanvas(self.fig)
        self.toolbar = NavigationToolbar(self.canvas, self)

        main_layout.addLayout(region_layout)
        main_layout.addWidget(self.toolbar)
        main_layout.addWidget(self.canvas)

        main_widget.setLayout(main_layout)
        self.setCentralWidget(main_widget)

        self.update_plot()


    def update_plot(self):
        selected_region = self.region_combo.currentText()
        data = self.df_lot if selected_region == 'All' else self.df_lot[self.df_lot['region'] == selected_region]

        wf_x = data['wf_x'].tolist()
        wf_y = data['wf_y'].tolist()

        variables = [
            'X_reg', 'Y_reg', 'mrc_fit_x', 'mrc_fit_y', 'X_reg_demrc', 'Y_reg_demrc',
            'MRC_X', 'MRC_Y', 'MRC_RX', 'MRC_RY', 'raw_x', 'raw_y', 'pred_x', 'pred_y',
            'residual_x', 'residual_y', 'psm_fit_x', 'psm_fit_y', 'residual_x_depsm',
            'residual_y_depsm', 'cpe_pred_x', 'cpe_pred_y', 'cpe_resi_x', 'cpe_resi_y',
            'ideal_psm_x', 'ideal_psm_y', 'delta_psm_x', 'delta_psm_y'
        ]
        data_vars = {var: data[var].tolist() for var in variables}

        # 서브플롯 레이아웃 동적 설정
        n_plots = 15  # 플롯 개수
        n_cols = 5    # 열 개수
        n_rows = -(-n_plots // n_cols)  # 행 개수 계산 (올림)

        self.fig.clear()
        self.fig.set_size_inches(15, 8)  # 전체 크기 설정
        axes = self.fig.subplots(n_rows, n_cols)  # 서브플롯 생성
        self.fig.subplots_adjust(wspace=0.4, hspace=0.4)  # 간격 조정

        # 수평선 및 수직선 계산
        step_pitch_x = self.df_lot['STEP_PITCH_X'].iloc[0]
        step_pitch_y = self.df_lot['STEP_PITCH_Y'].iloc[0]
        map_shift_x = self.df_lot['MAP_SHIFT_X'].iloc[0]
        map_shift_y = self.df_lot['MAP_SHIFT_Y'].iloc[0]
        start_left = -(step_pitch_x) / 2 + map_shift_x
        start_bottom = -(step_pitch_y) / 2 + map_shift_y
        max_die_x = int(self.df_lot['DieX'].max())
        min_die_x = int(self.df_lot['DieX'].min())
        max_die_y = int(self.df_lot['DieY'].max())
        min_die_y = int(self.df_lot['DieY'].min())

        self.vertical_lines = calculate_lines(start_left, step_pitch_x, max_die_x, min_die_x)
        self.horizontal_lines = calculate_lines(start_bottom, step_pitch_y, max_die_y, min_die_y)

        # 플롯 설정
        plot_configs = [
            {'dx': data_vars['X_reg'], 'dy': data_vars['Y_reg'], 'title': 'Pure_Raw(X_reg,Y_reg)'},
            {'dx': data_vars['mrc_fit_x'], 'dy': data_vars['mrc_fit_y'], 'title': 'K MRC Fitting'},
            {'dx': data_vars['X_reg_demrc'], 'dy': data_vars['Y_reg_demrc'], 'title': 'X_reg - KMRC'},
            {'dx': data_vars['MRC_X'], 'dy': data_vars['MRC_Y'], 'title': '-PSM + PointMRC'},
            {'dx': data_vars['MRC_RX'], 'dy': data_vars['MRC_RY'], 'title': 'PointMRC'},
            {'dx': data_vars['raw_x'], 'dy': data_vars['raw_y'], 'title': 'ADI RAW(X_reg_demrc - MRC_X)'},
            {'dx': data_vars['pred_x'], 'dy': data_vars['pred_y'], 'title': 'Fitting(WK,RK)'},
            {'dx': data_vars['residual_x'], 'dy': data_vars['residual_y'], 'title': 'Residual(WK,RK)'},
            {'dx': data_vars['psm_fit_x'], 'dy': data_vars['psm_fit_y'], 'title': 'PSM Input'},
            {'dx': data_vars['residual_x_depsm'], 'dy': data_vars['residual_y_depsm'], 'title': 'Residual - PSM'},
            {'dx': data_vars['cpe_pred_x'], 'dy': data_vars['cpe_pred_y'], 'title': 'CPE Fitting'},
            {'dx': data_vars['cpe_resi_x'], 'dy': data_vars['cpe_resi_y'], 'title': 'CPE Residual'},
            {'dx': data_vars['ideal_psm_x'], 'dy': data_vars['ideal_psm_y'], 'title': 'Absolute PSM'},
            {'dx': data_vars['delta_psm_x'], 'dy': data_vars['delta_psm_y'], 'title': 'Relative PSM'},
        ]

        self.quivers = []
        for i, config in enumerate(plot_configs):
            row, col = divmod(i, n_cols)
            ax = axes[row, col]
            quiver = plot_overlay(
                ax, wf_x, wf_y, config['dx'], config['dy'],
                self.vertical_lines, self.horizontal_lines,
                title=config['title'], show_labels=True
            )
            self.quivers.append((quiver, config['title']))

        # 남은 서브플롯 숨기기
        for j in range(len(plot_configs), n_rows * n_cols):
            row, col = divmod(j, n_cols)
            axes[row, col].axis('off')

        self.canvas.draw()

        # 기존 이벤트 연결 해제 및 재설정
        if self.cid is not None:
            self.canvas.mpl_disconnect(self.cid)
        self.cid = self.canvas.mpl_connect('button_press_event', self.on_click)


    def on_click(self, event):
        for quiver, title in self.quivers:
            if event.inaxes == quiver.axes and quiver.contains(event)[0]:
                selected_region = self.region_combo.currentText()
                data = self.df_lot if selected_region == 'All' else self.df_lot[self.df_lot['region'] == selected_region]

                wf_x = data['wf_x'].tolist()
                wf_y = data['wf_y'].tolist()
                dx = quiver.U
                dy = quiver.V

                enlarged_window = EnlargedPlotWindow(
                    wf_x, wf_y, dx, dy,
                    title=title + f' - {selected_region}',
                    v_lines=self.vertical_lines,
                    h_lines=self.horizontal_lines,
                    wafer_radius=Config.WAFER_RADIUS,
                    scale_factor=Config.SCALE_FACTOR
                )
                self.enlarged_plot_windows.append(enlarged_window)
                enlarged_window.show()
