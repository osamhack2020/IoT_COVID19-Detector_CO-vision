# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'ui_main_window_test02.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_Form(object):
    def setupUi(self, Form):
        Form.setObjectName("Form")
        Form.resize(1045, 677)
        self.horizontalLayout = QtWidgets.QHBoxLayout(Form)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.main_video = QtWidgets.QLabel(Form)
        self.main_video.setScaledContents(False)
        self.main_video.setObjectName("main_video")
        self.horizontalLayout_2.addWidget(self.main_video)
        self.thermal_video = QtWidgets.QLabel(Form)
        self.thermal_video.setScaledContents(False)
        self.thermal_video.setObjectName("thermal_video")
        self.horizontalLayout_2.addWidget(self.thermal_video)
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setObjectName("verticalLayout")
        self.label = QtWidgets.QLabel(Form)
        self.label.setScaledContents(False)
        self.label.setObjectName("label")
        self.verticalLayout.addWidget(self.label)
        self.label_2 = QtWidgets.QLabel(Form)
        self.label_2.setScaledContents(False)
        self.label_2.setObjectName("label_2")
        self.verticalLayout.addWidget(self.label_2)
        # self.label_3 = QtWidgets.QLabel(Form)
        # self.label_3.setScaledContents(False)
        # self.label_3.setObjectName("label_3")
        # self.verticalLayout.addWidget(self.label_3)
        self.horizontalLayout_2.addLayout(self.verticalLayout)
        self.horizontalLayout.addLayout(self.horizontalLayout_2)

        self.retranslateUi(Form)
        QtCore.QMetaObject.connectSlotsByName(Form)

    def retranslateUi(self, Form):
        _translate = QtCore.QCoreApplication.translate
        Form.setWindowTitle(_translate("Form", "Cam view"))
        self.main_video.setText(_translate("Form", "TextLabel"))
        self.thermal_video.setText(_translate("Form", "TextLabel"))
        self.label.setText(_translate("Form", "TextLabel"))
        self.label_2.setText(_translate("Form", "TextLabel"))
        #self.label_3.setText(_translate("Form", "TextLabel"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    Form = QtWidgets.QWidget()
    ui = Ui_Form()
    ui.setupUi(Form)
    Form.show()
    sys.exit(app.exec_())

