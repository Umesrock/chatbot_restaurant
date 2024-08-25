from PyQt5 import QtCore, QtGui, QtWidgets
import random
import json
import torch
from model import NeuralNet
from nltk_utils import bag_of_words, tokenize


class Ui_Form3(object):
    def setupUi(self, Form3):
        Form3.setObjectName("Form3")
        Form3.resize(934, 742)
        self.label = QtWidgets.QLabel(Form3)
        self.label.setGeometry(QtCore.QRect(0, 0, 934, 742))
        self.label.setText("")
        self.label.setPixmap(QtGui.QPixmap("252.jpg"))
        self.label.setScaledContents(True)
        self.label.setObjectName("label")
        self.label_3 = QtWidgets.QLabel(Form3)
        self.label_3.setGeometry(QtCore.QRect(290, 40, 431, 61))
        font = QtGui.QFont()
        font.setPointSize(32)
        font.setBold(True)
        font.setWeight(75)
        self.label_3.setFont(font)
        self.label_3.setStyleSheet("color: rgb(0, 255, 255);")
        self.label_3.setObjectName("label_3")
        self.label_4 = QtWidgets.QLabel(Form3)
        self.label_4.setGeometry(QtCore.QRect(60, 200, 281, 41))
        font = QtGui.QFont()
        font.setPointSize(26)
        font.setBold(True)
        font.setWeight(75)
        self.label_4.setFont(font)
        self.label_4.setStyleSheet("color: rgb(255, 255, 255);")
        self.label_4.setObjectName("label_4")

        self.label_5 = QtWidgets.QLabel(Form3)
        self.label_5.setGeometry(QtCore.QRect(60, 280, 481, 41))
        font = QtGui.QFont()
        font.setPointSize(26)
        font.setBold(True)
        font.setWeight(75)
        self.label_5.setFont(font)
        self.label_5.setStyleSheet("color: rgb(255, 255, 255);")
        self.label_5.setObjectName("label_5")

        self.lineEdit = QtWidgets.QLineEdit(Form3)
        self.lineEdit.setGeometry(QtCore.QRect(350, 200, 551, 41))
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(18)
        self.lineEdit.setFont(font)
        self.lineEdit.setObjectName("lineEdit")

        self.textEdit = QtWidgets.QTextEdit(Form3)
        self.textEdit.setGeometry(QtCore.QRect(350, 280, 551, 141))

        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(18)
        self.textEdit.setFont(font)

        # Enable word wrap for the text edit
        self.textEdit.setWordWrapMode(QtGui.QTextOption.WordWrap)

        self.nextButton = QtWidgets.QPushButton(Form3)
        self.nextButton.setGeometry(QtCore.QRect(360, 620, 211, 61))
        font = QtGui.QFont()
        font.setPointSize(21)
        font.setBold(True)
        font.setWeight(75)
        self.nextButton.setFont(font)
        self.nextButton.setStyleSheet("background-color: rgb(19,150, 160);\n"
                                      "border-radius:20px")
        self.nextButton.setObjectName("nextButton")
        self.nextButton.clicked.connect(self.processing_image)

        self.timer = QtCore.QTimer(Form3)
        self.timer.timeout.connect(self.check_for_response)

        self.retranslateUi(Form3)
        QtCore.QMetaObject.connectSlotsByName(Form3)

        # Initialize variables
        self.user_input = ""
        self.response_generated = False

    def processing_image(self):
        self.user_input = self.lineEdit.text()
        self.response_generated = False
        self.timer.start(1000)  # Start the timer to check for a response every second

    def check_for_response(self):
        if not self.response_generated:
            self.run_chatbot()
        else:
            self.timer.stop()

    def run_chatbot(self):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        with open('intents.json', 'r') as json_data:
            intents = json.load(json_data)

        FILE = "data.pth"
        data = torch.load(FILE)

        input_size = data["input_size"]
        hidden_size = data["hidden_size"]
        output_size = data["output_size"]
        all_words = data['all_words']
        tags = data['tags']
        model_state = data["model_state"]

        model = NeuralNet(input_size, hidden_size, output_size).to(device)
        model.load_state_dict(model_state)
        model.eval()

        bot_name = "RestBot"
        print("Let's chat! (type 'quit' to exit)")

        sentence = self.user_input
        if sentence == "quit":
            return

        sentence = tokenize(sentence)
        X = bag_of_words(sentence, all_words)
        X = X.reshape(1, X.shape[0])
        X = torch.from_numpy(X).to(device)

        output = model(X)
        _, predicted = torch.max(output, dim=1)

        tag = tags[predicted.item()]

        probs = torch.softmax(output, dim=1)
        prob = probs[0][predicted.item()]

        if prob.item() > 0.75:
            for intent in intents['intents']:
                if tag == intent["tag"]:
                    response = random.choice(intent['responses'])
                    self.update_line_edit(response)
                    self.response_generated = True
        else:
            print(f"{bot_name}: I do not understand...")
            self.update_line_edit("I do not understand...")
            self.response_generated = True

    def update_line_edit(self, response):
        self.textEdit.clear()  # Clear existing content
        new_text = response
        self.textEdit.setPlainText(new_text)

    def retranslateUi(self, Form3):
        _translate = QtCore.QCoreApplication.translate
        Form3.setWindowTitle(_translate("Form3", "Form"))
        self.label_3.setText(_translate("Form3", "LIVE CHAT"))
        self.label_4.setText(_translate("Form3", "You:"))
        self.label_5.setText(_translate("Form3", "Bot:"))
        self.nextButton.setText(_translate("Form3", "OK"))


if __name__ == "__main__":
    import sys

    app = QtWidgets.QApplication(sys.argv)
    Form3 = QtWidgets.QWidget()
    ui = Ui_Form3()
    ui.setupUi(Form3)
    Form3.show()
    sys.exit(app.exec_())
