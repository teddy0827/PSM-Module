from PyQt5.QtWidgets import QMainWindow, QWidget, QVBoxLayout, QLabel, QLineEdit, QListWidget, QPushButton, QMessageBox
from windows.plot_window import PlotWindow

class MainWindow(QMainWindow):
    def __init__(self, df):
        super().__init__()
        self.setWindowTitle("Unique ID Plot Viewer")
        self.setGeometry(100, 100, 400, 400)

        layout = QVBoxLayout()

        label = QLabel("Select a unique_id to view the plot:")
        layout.addWidget(label)

        self.search_box = QLineEdit()
        self.search_box.setPlaceholderText("Search unique_id")
        layout.addWidget(self.search_box)

        self.list_widget = QListWidget()
        layout.addWidget(self.list_widget)

        self.df = df
        self.unique_ids = sorted(df['UNIQUE_ID'].unique())
        self.list_widget.addItems(self.unique_ids)

        self.search_box.textChanged.connect(self.filter_unique_ids)

        button = QPushButton("Show Plot")
        button.clicked.connect(self.show_plot)
        layout.addWidget(button)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

    def filter_unique_ids(self, text):
        self.list_widget.clear()
        filtered_ids = [uid for uid in self.unique_ids if text.lower() in uid.lower()]
        self.list_widget.addItems(filtered_ids)

    def show_plot(self):
        selected_items = self.list_widget.selectedItems()
        if selected_items:
            unique_id = selected_items[0].text()
            self.plot_window = PlotWindow(unique_id, self.df)
            self.plot_window.show()
        else:
            QMessageBox.warning(self, "No Selection", "Please select a unique_id from the list.")
