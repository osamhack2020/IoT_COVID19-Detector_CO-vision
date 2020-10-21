import sys

# import some PyQt5 modules
from PyQt5.QtWidgets import QApplication
from PyQt5.QtWidgets import QWidget
from PyQt5.QtGui import QImage
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import QTimer

# import Opencv module
import cv2

from ui_main_window_test03 import *

class MainWindow(QWidget):
    # class constructor
    def __init__(self):
        # call QWidget constructor
        super().__init__()
        self.ui = Ui_Form()
        self.ui.setupUi(self)

        # create a timer
        self.timer = QTimer()
        # set timer timeout callback function
        self.timer.timeout.connect(self.viewCam)
        self.timer.timeout.connect(self.viewThermalCam)
        self.cap =  cv2.VideoCapture('imgs/junha_video.mp4') # 여기는 캠 0을 넣으면 됨
        self.capthermal = cv2.VideoCapture('imgs/junha_video.mp4') #여기는 열화상 카메라 넣으면 됨
        # start timer
        self.timer.start(20)



    # view camera
    def viewCam(self):
        # read image in BGR format
        ret, image = self.cap.read()
        image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        image = cv2.resize(image, dsize=(0, 0), fx=0.2, fy=0.2, interpolation=cv2.INTER_LINEAR)
        # convert image to RGB format
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # get image infos
        height, width, channel = image.shape
        step = channel * width
        # create QImage from image
        qImg = QImage(image.data, width, height, step, QImage.Format_RGB888)
        # show image in main_video
        self.ui.main_video.setPixmap(QPixmap.fromImage(qImg))
    def viewThermalCam(self):
        # read image in BGR format
        ret, image = self.capthermal.read()
        image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        image = cv2.resize(image, dsize=(0, 0), fx=0.2, fy=0.2, interpolation=cv2.INTER_LINEAR)
        # convert image to RGB format
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # get image infos
        height, width, channel = image.shape
        step = channel * width
        # create QImage from image
        qImg = QImage(image.data, width, height, step, QImage.Format_RGB888)
        # show image in thermal_video
        self.ui.thermal_video.setPixmap(QPixmap.fromImage(qImg))
    # start/stop timer
    # def controlTimer(self):
    #     # if timer is stopped
    #     if not self.timer.isActive():
    #         # create video capture
    #         self.cap =  cv2.VideoCapture('imgs/junha_video.mp4')
    #         self.capthermal = cv2.VideoCapture('imgs/junha_video.mp4')
    #         # start timer
    #         self.timer.start(20)
    #         # update control_bt text
    #         self.ui.control_bt.setText("Stop")
    #     # if timer is started
    #     else:
    #         # stop timer
    #         self.timer.stop()
    #         # release video capture
    #         self.cap.release()
    #         # update control_bt text
    #         self.ui.control_bt.setText("Start")


if __name__ == '__main__':
    app = QApplication(sys.argv)

    # create and show mainWindow
    mainWindow = MainWindow()
    mainWindow.show()

    sys.exit(app.exec_())
