import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QTextEdit
from PyQt5.QtCore import QTextStream, Qt
from PyQt5.QtGui import QTextCursor
from PyQt5.uic import loadUi
import io

from finetuner import train


class TextRedirector(io.TextIOBase):
    def __init__(self, text_widget):
        self.text_widget = text_widget

    def write(self, s):
        self.text_widget.moveCursor(QTextCursor.End)
        self.text_widget.insertPlainText(s)

    def flush(self):
        pass


class MyWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        loadUi("main_window.ui", self)  # Load the .ui file

        self.setWindowTitle("HF Fine-Tune")
        self.pushButton_input.clicked.connect(self.select_input)
        self.pushButton_output.clicked.connect(self.select_output)
        self.pushButton_data.clicked.connect(self.select_data)
        self.pushButton_train.clicked.connect(self.train)
        self.pushButton_evaluate.clicked.connect(self.evaluate)
        self.pushButton_inference.clicked.connect(self.inference)

        # Set output to textEdit_log
        sys.stdout = TextRedirector(self.textEdit_log)

        # Set app icon to generic gear
        self.setWindowFlags(self.windowFlags() | Qt.Window)
        self.setWindowIcon(self.style().standardIcon(1))



    def select_input(self):
        input_path = QFileDialog.getExistingDirectory(self, "Select Input Folder")
        if input_path:
            self.lineEdit_input.setText(input_path)

    def select_output(self):
        output_path = QFileDialog.getExistingDirectory(self, "Select Output Folder")
        if output_path:
            self.lineEdit_output.setText(output_path)

    def select_data(self):
        data_path = QFileDialog.getExistingDirectory(self, "Select Data Folder")
        if data_path:
            self.lineEdit_data.setText(data_path)

    def train(self):
        if not self.lineEdit_input.text() or not self.lineEdit_output.text() or not self.lineEdit_data.text():
            self.statusbar.showMessage("Input and/or output paths are empty")
            return
        
        print("Training on", "GPU" if self.checkBox_gpu.isChecked() else "CPU")
        train(self.checkBox_gpu.isChecked(), self.lineEdit_input.text(), self.lineEdit_output.text(), self.lineEdit_data.text())

    def evaluate(self):
        if not self.lineEdit_input.text() or not self.lineEdit_output.text() or not self.lineEdit_data.text():
            self.statusbar.showMessage("Input and/or output paths are empty")
            return
        print("Evaluating on", "GPU" if self.checkBox_gpu.isChecked() else "CPU")
        print("Not implemented yet")

    def inference(self):
        if not self.lineEdit_input.text() or not self.lineEdit_data.text():
            self.statusbar.showMessage("Input and/or output paths are empty")
            return
        print("Inference on", "GPU" if self.checkBox_gpu.isChecked() else "CPU")
        print("Not implemented yet")



if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MyWindow()
    window.show()
    sys.exit(app.exec_())
