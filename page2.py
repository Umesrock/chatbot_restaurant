# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'page2.ui'
#
# Created by: PyQt5 UI code generator 5.15.4
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets
from page3 import Ui_Form3


class Ui_Form22(object):
    def setupUi(self, Form22):
        Form22.setObjectName("Form22")
        Form22.resize(934, 742)
        self.label = QtWidgets.QLabel(Form22)
        self.label.setGeometry(QtCore.QRect(0, 0, 934, 742))
        self.label.setText("")
        self.label.setPixmap(QtGui.QPixmap("33.jpg"))
        self.label.setScaledContents(True)
        self.label.setObjectName("label")
        self.label_2 = QtWidgets.QLabel(Form22)
        self.label_2.setGeometry(QtCore.QRect(350, 20, 501, 91))
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(40)
        font.setBold(True)
        font.setWeight(75)
        self.label_2.setFont(font)
        self.label_2.setStyleSheet("color: rgb(0,255,255);")
        self.label_2.setObjectName("label_2")
        self.label_3 = QtWidgets.QLabel(Form22)
        self.label_3.setGeometry(QtCore.QRect(180, 280, 361, 51))
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(24)
        font.setBold(True)
        font.setWeight(75)
        self.label_3.setFont(font)
        self.label_3.setStyleSheet("color: rgb(0,255,255);")
        self.label_3.setObjectName("label_3")
        self.label_4 = QtWidgets.QLabel(Form22)
        self.label_4.setGeometry(QtCore.QRect(180, 190, 361, 41))
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(24)
        font.setBold(True)
        font.setWeight(75)
        self.label_4.setFont(font)
        self.label_4.setStyleSheet("color: rgb(0,255,255);")
        self.label_4.setObjectName("label_4")
        self.lineEdit = QtWidgets.QLineEdit(Form22)
        self.lineEdit.setGeometry(QtCore.QRect(550, 190, 251, 41))
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(24)
        self.lineEdit.setFont(font)
        self.lineEdit.setObjectName("lineEdit")
        self.lineEdit_2 = QtWidgets.QLineEdit(Form22)
        self.lineEdit_2.setGeometry(QtCore.QRect(550, 290, 251, 41))
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(24)
        self.lineEdit_2.setFont(font)
        self.lineEdit_2.setEchoMode(QtWidgets.QLineEdit.Password)
        self.lineEdit_2.setObjectName("lineEdit_2")
        self.pushButton = QtWidgets.QPushButton(Form22)
        self.pushButton.setGeometry(QtCore.QRect(570, 660, 271, 51))
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(19)
        font.setBold(True)
        font.setWeight(75)
        self.pushButton.setFont(font)
        self.pushButton.setStyleSheet("border-radius:20px;\n"
"background-color: rgb(19,150, 160);")
        self.pushButton.setObjectName("pushButton")
        self.pushButton.clicked.connect(self.next_page)

        self.lineEdit_6 = QtWidgets.QLineEdit(Form22)
        self.lineEdit_6.setGeometry(QtCore.QRect(360, 450, 381, 51))
        self.lineEdit_6.setStyleSheet("background-color: rgba(0, 0, 0,0);\n"
                                    "border:none;\n"
                                    "border-bottom:2px solid rgba(255,255,255,255);\n"
                                    "color: rgb(255, 255, 255);\n"
                                    "paddin-bottom:7px;\n"
                                    "font: 14pt \"MS Shell Dlg 2\";\n"
                                    "")
        self.lineEdit_6.setObjectName("lineEdit_6")
        self.lineEdit_21 = QtWidgets.QLineEdit(Form22)
        self.lineEdit_21.setGeometry(QtCore.QRect(360, 500, 381, 51))
        self.lineEdit_21.setStyleSheet("background-color: rgba(0, 0, 0,0);\n"
                                      "border:none;\n"
                                      "border-bottom:2px solid rgba(255,255,255,255);\n"
                                      "color: rgb(255,255, 255);\n"
                                      "paddin-bottom:7px;\n"
                                      "font: 14pt \"MS Shell Dlg 2\";\n"
                                      "")
        self.lineEdit_21.setObjectName("lineEdit_21")

        self.retranslateUi(Form22)
        QtCore.QMetaObject.connectSlotsByName(Form22)
        Form22.setWindowTitle("Admin login")


    def retranslateUi(self, Form22):
        _translate = QtCore.QCoreApplication.translate
        Form22.setWindowTitle(_translate("Form2", "Form"))
        self.label_2.setText(_translate("Form2", "LOG IN "))
        self.label_3.setText(_translate("Form2", "PASSWORD:"))
        self.label_4.setText(_translate("Form2", "USER_NAME:"))
        self.pushButton.setText(_translate("Form2", "LOG IN"))


    def next_page(self):
        n1 = self.lineEdit.text()
        n2 = self.lineEdit_2.text()
        if n1 == 'food':
            print("correct username")
            self.lineEdit_6.setText("correct username")
            if n2 == '123':
                print("correct password")
                self.lineEdit_21.setText("correct password")
                self.Form3 = QtWidgets.QMainWindow()
                self.ui = Ui_Form3()
                self.ui.setupUi(self.Form3)
                self.Form3.show()
            else:
                print("wrong password")
                self.lineEdit_21.setText("wrong password")
        else:
            print("wrong username")
            self.lineEdit_6.setText("wrong username")
if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    Form22 = QtWidgets.QWidget()
    ui = Ui_Form22()
    ui.setupUi(Form22)
    Form22.show()
    sys.exit(app.exec_())
