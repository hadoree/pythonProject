import sys, os, time, cv2
from socket import *
from multiprocessing import Process
from datetime import datetime
import numpy as np
import pandas as pd
import pymysql as pm
import mediapipe as mp
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.xception import preprocess_input
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5 import uic

ui = uic.loadUiType("untitled.ui")[0]

class Main(QMainWindow, ui):
    def __init__(self):
        QMainWindow.__init__(self)
        self.setupUi(self)
        self.pushButton.clicked.connect(self.upload_file)
        self.pushButton_2.clicked.connect(self.check_image)
        image_path = ""

    def upload_file(self):
        filter = 'Image(*.png *.jpg *.jpeg *.PNG bmp ) (.png *.jpg *.jpeg *bmp *.PNG)'
        self.image_path = QFileDialog.getOpenFileName(self, '파일 선택', filter=filter)
        self.image_path = self.image_path[0]
        image = QPixmap()
        image.load(self.image_path)
        self.label.setPixmap(image)

    def check_image(self):
        image = cv2.imread(self.image_path)
        image = cv2.resize(image, (400, 500))
        # # 입력 파일 지정하기
        # image_file = "mini.png"

        # 캐스케이드 파일의 경로 지정하기
        cascade_file = "haarcascade_smile.xml"
        # 이미지 읽어들이기
        image = cv2.imread(self.image_path)
        # 그레이스케일로 변환하기
        image_gs = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # 얼굴 인식 특징 파일 읽어들이기
        cascade = cv2.CascadeClassifier(cascade_file)
        # 얼굴인식 실행하기
        face_list = cascade.detectMultiScale(image_gs,
                                             scaleFactor=1.1,
                                             minNeighbors=1,
                                             minSize=(150,150))
        if len(face_list) > 0:
            # 인식한 부분 표시하기
            print(face_list)
            color = (0, 0, 255)
            for face in face_list:
                x, y, w, h = face
                cv2.rectangle(image, (x, y), (x+w, y+h), color, thickness=8)
                # 파일로 출력하기
            cv2.imwrite("smile_output.PNG", image)

            self.msg = QMessageBox()
            self.msg.setIcon(QMessageBox.Information)
            self.msg.setStandardButtons(QMessageBox.Ok)
            self.msg.setText("웃는 사진입니다!")
            self.msg.exec_()
        else:
            print("no face")
            self.msg = QMessageBox()
            self.msg.setIcon(QMessageBox.Information)
            self.msg.setStandardButtons(QMessageBox.Ok)
            self.msg.setText("웃지 않는 사진입니다!")
            self.msg.exec_()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    start = Main()
    start.show()
    app.exec_()