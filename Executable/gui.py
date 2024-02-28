import sys
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QVBoxLayout, QLabel, QFileDialog, QMessageBox
import requests

class PhotoSenderGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle('Photo Sender')
        self.setGeometry(100, 100, 200, 100)

        layout = QVBoxLayout()

        self.label = QLabel('Select a photo to send', self)
        layout.addWidget(self.label)

        btn = QPushButton('Choose Photo', self)
        btn.clicked.connect(self.openFileNameDialog)
        layout.addWidget(btn)

        self.setLayout(layout)

    def openFileNameDialog(self):
        options = QFileDialog.Options()
        fileName, _ = QFileDialog.getOpenFileName(self, "QFileDialog.getOpenFileName()", "", "All Files (*);;JPEG (*.jpg;*.jpeg);;PNG (*.png)", options=options)
        if fileName:
            self.sendPhoto(fileName)

    def sendPhoto(self, file_path):
        url = 'http://localhost:5000/verify-image'
        try:
            files = {'img1': open(file_path, 'rb')}
            response = requests.post(url, files=files)
            if response.status_code == 200:
                self.showMessage("Verified")
            elif response.status_code == 500:
                self.showMessage("Unverified")
            else:
                self.showMessage(f"Error: {response.status_code}")
        except requests.exceptions.RequestException as e:
            self.showMessage("Error: No response")

    def showMessage(self, message):
        msgBox = QMessageBox()
        msgBox.setIcon(QMessageBox.Information)
        msgBox.setText(message)
        msgBox.setWindowTitle("Photo Upload Status")
        msgBox.setStandardButtons(QMessageBox.Ok)
        msgBox.exec_()
if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = PhotoSenderGUI()
    ex.show()
    sys.exit(app.exec_())
