# import cv2
# import sys
# from PyQt5 import QtCore
# from PyQt5 import QtWidgets
# from PyQt5 import QtGui


# class ShowVideo(QtCore.QObject):

#     flag = 0

#     camera = cv2.VideoCapture('imgs/junha_video.mp4')
    

#     ret, image = camera.read()
#     image = cv2.resize(image, dsize=(0, 0), fx=0.3, fy=0.3, interpolation=cv2.INTER_AREA)
#     height, width = image.shape[:2]

#     VideoSignal1 = QtCore.pyqtSignal(QtGui.QImage)
#     VideoSignal2 = QtCore.pyqtSignal(QtGui.QImage)

#     def __init__(self, parent=None):
#         super(ShowVideo, self).__init__(parent)

#     @QtCore.pyqtSlot()
#     def startVideo(self):
#         global image
#         run_video = True
#         while run_video:
#             ret, image = self.camera.read()
#             image = cv2.resize(image, dsize=(0, 0), fx=0.3, fy=0.3, interpolation=cv2.INTER_AREA)
#             color_swapped_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

#             qt_image1 = QtGui.QImage(color_swapped_image.data,
#                                     self.width,
#                                     self.height,
#                                     color_swapped_image.strides[0],
#                                     QtGui.QImage.Format_RGB888)
#             self.VideoSignal1.emit(qt_image1)


            

#             loop = QtCore.QEventLoop()
#             QtCore.QTimer.singleShot(25, loop.quit) #25 ms
#             loop.exec_()


# class ImageViewer(QtWidgets.QWidget):
#     def __init__(self, parent=None):
#         super(ImageViewer, self).__init__(parent)
#         self.image = QtGui.QImage()
#         self.setAttribute(QtCore.Qt.WA_OpaquePaintEvent)

#     def paintEvent(self, event):
#         painter = QtGui.QPainter(self)
#         painter.drawImage(0, 0, self.image)
#         self.image = QtGui.QImage()

#     def initUI(self):
#         self.setWindowTitle('Test')

#     @QtCore.pyqtSlot(QtGui.QImage)
#     def setImage(self, image):
#         if image.isNull():
#             print("Viewer Dropped frame!")

#         self.image = image
#         if image.size() != self.size():
#             self.setFixedSize(image.size())
#         self.update()


# if __name__ == '__main__':
#     app = QtWidgets.QApplication(sys.argv)


#     thread = QtCore.QThread()
#     thread.start()
#     vid = ShowVideo()
#     vid.moveToThread(thread)

#     image_viewer1 = ImageViewer()
#     image_viewer2 = ImageViewer()

#     vid.VideoSignal1.connect(image_viewer1.setImage)
#     push_button1 = QtWidgets.QPushButton('Start')
#     push_button1.clicked.connect(vid.startVideo)

#     vertical_layout = QtWidgets.QVBoxLayout()
#     horizontal_layout = QtWidgets.QHBoxLayout()
#     horizontal_layout.addWidget(image_viewer1)
#     vertical_layout.addLayout(horizontal_layout)
#     vertical_layout.addWidget(push_button1)

#     layout_widget = QtWidgets.QWidget()
#     layout_widget.setLayout(vertical_layout)

#     main_window = QtWidgets.QMainWindow()
#     main_window.setCentralWidget(layout_widget)
#     main_window.show()
#     sys.exit(app.exec_())
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
import sys, cv2, numpy, time

class Example(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("cam_exam")
        self.setGeometry(150,150,650,540)
        self.iniUI()        
    
    def iniUI(self):
        self.cpt = cv2.VideoCapture('imgs/junha_video.mp4')
        self.fps = 24
        self.sens = 300
        _, self.img_o = self.cpt.read()
        self.img_o = cv2.cvtColor(self.img_o, cv2.COLOR_RGB2GRAY)
        #cv2.imwrite("img_o.jpg", self.img_o)

        self.cnt = 0

        self.frame = QLabel(self)
        self.frame.resize(640,480)
        self.frame.setScaledContents(True)
        self.frame.move(5,5)

        self.btn_on = QPushButton("On", self)
        self.btn_on.resize(100,25)
        self.btn_on.move(5,490)
        self.btn_on.clicked.connect(self.start)

        self.btn_off = QPushButton("Off", self)
        self.btn_off.resize(100,25)
        self.btn_off.move(5+100+5,490)
        self.btn_off.clicked.connect(self.stop)

        self.prt = QLabel(self)            
        self.prt.resize(200,25)
        self.prt.move(5+105+105, 490)

        self.sldr = QSlider(Qt.Horizontal, self)
        self.sldr.resize(100, 25)
        self.sldr.move(5+105+105+200, 490)
        self.sldr.setMinimum(1)
        self.sldr.setMaximum(30)
        self.sldr.setValue(24)
        self.sldr.valueChanged.connect(self.setFps)
        
        self.sldr1 = QSlider(Qt.Horizontal, self)
        self.sldr1.resize(100, 25)
        self.sldr1.move(5+105+105+200+105, 490)
        self.sldr1.setMinimum(50)
        self.sldr1.setMaximum(500)
        self.sldr1.setValue(300)
        self.sldr1.valueChanged.connect(self.setSens)
        self.show()

    def setFps(self):
        self.fps = self.sldr.value()
        self.prt.setText("FPS" + str(self.fps) + "로 조정")
        self.timer.stop()
        self.timer.start(1000/self.fps)

    def setSens(self):
        self.sens = self.sldr.value()
        self.prt.setText("감도" + str(self.sens) + "로 조정")

    def start(self):
        self.timer = QTimer()
        self.timer.timeout.connect(self.nextFrameSlot)
        self.timer.start(1000/self.fps)

    def nextFrameSlot(self):           
        success, cam = self.cpt.read()
        if success:
            cam = cv2.cvtColor(cam, cv2.COLOR_BGR2RGB)
            self.img_p = cv2.cvtColor(cam, cv2.COLOR_RGB2GRAY)
            #cv2.imwrite('img_p.jpg', self.img_p)
            self.compare(self.img_o, self.img_p)
            self.img_o = self.img_p.copy()
            img = QImage(cam, cam.shape[1], cam.shape[0], QImage.Format_RGB888)
            pix = QPixmap.fromImage(img)
            self.frame.setPixmap(pix)

    def stop(self):
        self.frame.setPixmap(QPixmap.fromImage(QImage()))
        self.timer.stop()

    def compare(self, img_o, img_p):
        err = numpy.sum((img_o.astype("float") - img_p.astype("float")) ** 2)
        err /= float(img_o.shape[0] * img_p.shape[1])
        if err >= self.sens:
            t = time.localtime()
            self.prt.setText("{}-{}-{} {}:{}:{} 움직임 감지".format(t.tm_year, t.tm_mon, t.tm_mday, t.tm_hour, t.tm_min, t.tm_sec))
        
if __name__ == "__main__":
    app = QApplication(sys.argv)
    ex = Example()
    sys.exit(app.exec_())