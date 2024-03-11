import sys
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QVBoxLayout, QLabel, QFileDialog, QMessageBox
import requests
import sounddevice as sd
from scipy.io.wavfile import write

class PhotoSenderGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle('Authenticate')
        self.setGeometry(200, 200, 300, 200)

        layout = QVBoxLayout()

        self.label = QLabel('Select a photo to send', self)
        layout.addWidget(self.label)

        btn = QPushButton('Choose Photo', self)
        btn.clicked.connect(self.openFileNameDialog)
        layout.addWidget(btn)

        btn_record = QPushButton('Record Message', self)
        btn_record.clicked.connect(self.recordMessage)
        layout.addWidget(btn_record)

        self.setLayout(layout)

    def openFileNameDialog(self):
        options = QFileDialog.Options()
        fileName, _ = QFileDialog.getOpenFileName(self, "QFileDialog.getOpenFileName()", "", "All Files (*);;JPEG (*.jpg;*.jpeg);;PNG (*.png)", options=options)
        if fileName:
            self.sendPhoto(fileName)

    def recordMessage(self):
        fs = 44100
        seconds = 5

        myrecording = sd.rec(int(seconds * fs), samplerate=fs, channels=2)
        sd.wait()
        write('output.wav', fs, myrecording)
        self.sendAudio('output.wav')

    def sendAudio(self, file_path):
        url = 'http://localhost:5000/verify-audio'
        try:
            files = {'audio': open(file_path, 'rb')}
            response = requests.post(url, files=files)
            if response.status_code == 200:
                self.showMessage("Audio sent successfully")
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
