# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'page1.ui'
#
# Created by: PyQt5 UI code generator 5.15.4
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets
from page2 import Ui_Form22

class Ui_Form(object):
    def setupUi(self, Form):
        Form.setObjectName("Form")
        Form.resize(934, 742)
        self.label = QtWidgets.QLabel(Form)
        self.label.setGeometry(QtCore.QRect(0, 0, 941, 751))
        self.label.setText("")
        self.label.setPixmap(QtGui.QPixmap("11.jpg"))
        self.label.setScaledContents(True)
        self.label.setObjectName("label")
        self.label_2 = QtWidgets.QLabel(Form)
        self.label_2.setGeometry(QtCore.QRect(80, 30, 841, 51))
        font = QtGui.QFont()
        font.setPointSize(32)
        font.setBold(True)
        font.setWeight(75)
        self.label_2.setFont(font) 
        self.label_2.setStyleSheet("color: rgb(0,255,255);")
        self.label_2.setObjectName("label_2")

        self.studen1 = QtWidgets.QLabel(Form)
        self.studen1.setGeometry(QtCore.QRect(410, 230, 411, 41))
        font = QtGui.QFont()
        font.setPointSize(26)
        font.setBold(True)
        font.setWeight(75)
        self.studen1.setFont(font)
        self.studen1.setStyleSheet("color: rgb(255, 255, 255);")
        self.studen1.setObjectName("studen1")
        self.student2 = QtWidgets.QLabel(Form)
        self.student2.setGeometry(QtCore.QRect(410, 300, 451, 31))
        font = QtGui.QFont()
        font.setPointSize(26)
        font.setBold(True)
        font.setWeight(75)
        self.student2.setFont(font)
        self.student2.setStyleSheet("color: rgb(255, 255, 255);")
        self.student2.setObjectName("student2")
        self.student3 = QtWidgets.QLabel(Form)
        self.student3.setGeometry(QtCore.QRect(410, 370, 421, 41))
        font = QtGui.QFont()
        font.setPointSize(26)
        font.setBold(True)
        font.setWeight(75)
        self.student3.setFont(font)
        self.student3.setStyleSheet("color: rgb(255, 255, 255);")
        self.student3.setObjectName("student3")
        # self.student4 = QtWidgets.QLabel(Form)
        # self.student4.setGeometry(QtCore.QRect(410, 440, 421, 41))
        # font = QtGui.QFont()
        # font.setPointSize(26)
        # font.setBold(True)
        # font.setWeight(75)
        # self.student4.setFont(font)
        # self.student4.setStyleSheet("color: rgb(0, 0, 0);")
        # self.student4.setObjectName("student4")
        self.nextButton = QtWidgets.QPushButton(Form)
        self.nextButton.setGeometry(QtCore.QRect(680, 660, 211, 61))
        font = QtGui.QFont()
        font.setPointSize(21)
        font.setBold(True)
        font.setWeight(75)
        self.nextButton.setFont(font)
        self.nextButton.setStyleSheet("background-color: rgb(19,150, 160);\n"
"border-radius:20px")
        self.nextButton.setObjectName("nextButton")
        self.nextButton.clicked.connect(self.next_page)
        self.retranslateUi(Form)
        QtCore.QMetaObject.connectSlotsByName(Form)

    def retranslateUi(self, Form):
        _translate = QtCore.QCoreApplication.translate
        Form.setWindowTitle(_translate("Form", "Form"))
        self.label_2.setText(_translate("Form", " WELCOME TO OUR HOTEL "))

        self.studen1.setText(_translate("Form", "Umesh"))
        self.student2.setText(_translate("Form", "Thowhith"))
        self.student3.setText(_translate("Form", "Venkatesh"))
        # self.student4.setText(_translate("Form", ""))
        self.nextButton.setText(_translate("Form", "Next"))



    def next_page(self):
        self.Form22 = QtWidgets.QMainWindow()
        self.ui = Ui_Form22()
        self.ui.setupUi(self.Form22)
        self.Form22.show()

if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    Form = QtWidgets.QWidget()
    ui = Ui_Form()
    ui.setupUi(Form)
    Form.show()
    sys.exit(app.exec_())
