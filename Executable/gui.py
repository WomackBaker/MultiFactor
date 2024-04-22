import sys
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QVBoxLayout, QLabel, QFileDialog, QMessageBox
import requests
import sounddevice as sd
from scipy.io.wavfile import write

# Sends jpeg for facial recognition and sends wav for voice recognition
class PhotoSenderGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle('Authenticate')
        self.setGeometry(200, 200, 300, 200)

        layout = QVBoxLayout()

        btn_record = QPushButton('Random', self)
        #TODO btn_record.clicked.connect(self.openFileNameDialog2)
        layout.addWidget(btn_record)

        btn = QPushButton('Facial Recognition', self)
        btn.clicked.connect(self.openFileNameDialog)
        layout.addWidget(btn)

        btn_record = QPushButton('Voice Recognition', self)
        btn_record.clicked.connect(self.openFileNameDialog2)
        layout.addWidget(btn_record)

        btn_record = QPushButton('SMS', self)
        #TODO btn_record.clicked.connect(self.openFileNameDialog2)
        layout.addWidget(btn_record)

        btn_record = QPushButton('Password', self)
        #TODO btn_record.clicked.connect(self.openFileNameDialog2)
        layout.addWidget(btn_record)

        self.setLayout(layout)

    def openFileNameDialog(self):
        options = QFileDialog.Options()
        fileName, _ = QFileDialog.getOpenFileName(self, "QFileDialog.getOpenFileName()", "", "All Files (*);;JPEG (*.jpg;*.jpeg);;PNG (*.png)", options=options)
        if fileName:
            self.sendPhoto(fileName)
    def openFileNameDialog2(self):
        options = QFileDialog.Options()
        fileName, _ = QFileDialog.getOpenFileName(self, "QFileDialog.getOpenFileName()", "", "All Files (*);;JPEG (*.jpg;*.jpeg);;PNG (*.png)", options=options)
        if fileName:
            self.sendAudio(fileName)

    def sendAudio(self, file_path):
        url = 'http://localhost:5000//verify_voice'
        try:
            files = {'voice': open(file_path, 'rb')}
            data = {'name': 'Baker'}
            response = requests.post(url, files=files, data=data)
            if response.status_code == 200:
                self.showMessage("Verified")
            elif response.status_code == 500:
                self.showMessage("Error")
            elif response.status_code == 400:
                self.showMessage("Unverified")
            else:
                self.showMessage(f"Error: {response.status_code}")
        except requests.exceptions.RequestException as e:
            self.showMessage("Error: No response")

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
