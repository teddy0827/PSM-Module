import matplotlib.pyplot as plt
from PyQt5.QtWidgets import QMainWindow, QWidget, QVBoxLayout
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from plot_helpers import plot_overlay

class EnlargedPlotWindow(QMainWindow):
    def __init__(self, x, y, u, v, title, v_lines, h_lines, wafer_radius, scale_factor):
        super().__init__()
        self.setWindowTitle(title)
        self.setGeometry(100, 100, 800, 800)

        fig, ax = plt.subplots(figsize=(8, 8))
        plot_overlay(ax, x, y, u, v, v_lines, h_lines, wafer_radius, title, scale_factor)

        self.canvas = FigureCanvas(fig)
        self.toolbar = NavigationToolbar(self.canvas, self)

        layout = QVBoxLayout()
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)
