# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'main_ui.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(921, 578)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.main_video = QtWidgets.QLabel(self.centralwidget)
        self.main_video.setGeometry(QtCore.QRect(50, 70, 321, 381))
        self.main_video.setText("")
        self.main_video.setPixmap(QtGui.QPixmap("NO_Mask_File/0_No_Mask99%_1.jpg"))
        self.main_video.setScaledContents(True)
        self.main_video.setObjectName("main_video")
        self.thermal_video = QtWidgets.QLabel(self.centralwidget)
        self.thermal_video.setGeometry(QtCore.QRect(380, 70, 321, 381))
        self.thermal_video.setText("")
        self.thermal_video.setPixmap(QtGui.QPixmap("NO_Mask_File/0_No_Mask98%_2.jpg"))
        self.thermal_video.setScaledContents(True)
        self.thermal_video.setObjectName("thermal_video")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 921, 21))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())

