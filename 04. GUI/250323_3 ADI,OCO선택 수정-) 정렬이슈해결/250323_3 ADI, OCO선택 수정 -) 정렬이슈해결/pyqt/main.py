import sys
import pandas as pd
from PyQt5.QtWidgets import QApplication, QFileDialog
from windows.main_window import MainWindow

def main():
    app = QApplication(sys.argv)
    file_dialog = QFileDialog()
    file_path, _ = file_dialog.getOpenFileName(None, "Select CSV File", "", "CSV Files (*.csv)")
    if not file_path:
        sys.exit(0)

    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        print(f"Error loading file: {e}")
        sys.exit(1)

    main_window = MainWindow(df)
    main_window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
