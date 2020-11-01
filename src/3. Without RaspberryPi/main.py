# Data Set을 만들어서 학습.
import flir_image_extractor
from flirpy.camera.lepton import Lepton
import cv2
import numpy as np
import os
import io
from google.cloud import vision
import telepot
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import load_model

# import some PyQt5 modules
from PyQt5.QtWidgets import QApplication
from PyQt5.QtWidgets import QWidget
from PyQt5.QtGui import QImage
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import QTimer

from ui_main_window import *

from collections import deque

# telegram API bot = co_vision_bot
token = '1130712531:AAE3W0J9Y3s2opGvE_c8My8e96-vhqlLAGE'
mc = '1314303321'
bot = telepot.Bot(token)
# 구글 API 설정
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = r'ServiceAccountToken.json'
client = vision.ImageAnnotatorClient()

facenet = cv2.dnn.readNet('MaskDetection/models/deploy.prototxt', 'MaskDetection/models/res10_300x300_ssd_iter_140000.caffemodel')
# FaceDetector 모델 > OpenCv의 DNN
model = load_model('MaskDetection/models/mask_detector.model')
# MaskDetector 모델 > Keras 모델

thermal_camera = Lepton()
fir = flir_image_extractor.FlirImageExtractor()


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
        #
        # self.timer.timeout.connect(self.viewThermalCam)
        self.cap = cv2.VideoCapture(0)  # 0으로 하면 웹캠 실시간으로 나옴
        self.capthermal = cv2.VideoCapture(1)
        self.number = 0
        self.dq = deque()
        self.textdq = deque([])
        # start timer
        self.timer.start(20)
        image = cv2.imread('Co-Vision_Logo.png')
        image = cv2.resize(image, dsize=(0, 0), fx=0.5, fy=0.7, interpolation=cv2.INTER_LINEAR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # get image infos
        height, width, channel = image.shape
        step = channel * width
        # create QImage from image
        qImg = QImage(image.data, width, height, step, QImage.Format_RGB888)
        # show image in main_video
        self.ui.label_3.setPixmap(QPixmap.fromImage(qImg))
        self.prevTime = 0
        self.Final_Text = ""
        self.temperature = 0
        self.nomask = 0

    def put_img_to_labels(self, dq):

        image = self.dq[0]
        image = cv2.resize(image, dsize=(0, 0), fx=0.2, fy=0.2, interpolation=cv2.INTER_LINEAR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width, channel = image.shape
        step = channel * width
        qImg = QImage(image.data, width, height, step, QImage.Format_RGB888)
        self.ui.label.setPixmap(QPixmap.fromImage(qImg))

    def label_2_text(self, textdq):
        TEXT = ""
        if len(self.textdq) > 10:
            textdq.pop()
        for text in textdq:
            TEXT += text + '\n'
        self.ui.label_2.setText(TEXT)

    def find_name_and_display(self, IMAGE_FILE, x1, x2, result_img, color):
        with io.open(IMAGE_FILE, 'rb') as image_file:
            content = image_file.read()

        image = vision.Image(content=content)
        response = client.document_text_detection(image=image)
        Final_Text = ""
        for data in response.text_annotations:
            xx1 = data.bounding_poly.vertices[0].x - 60  # 박스가 너무 오른쪽으로 나옴 그래서 수정함.
            yy1 = data.bounding_poly.vertices[0].y
            xx2 = data.bounding_poly.vertices[2].x
            yy2 = data.bounding_poly.vertices[2].y + 20
            if xx1 > (x1 + x2) // 2 or xx2 > (x1 + x2) // 2:
                continue
            for x in data.description:
                if ord('가') <= ord(x) <= ord('힣'):
                    cv2.rectangle(result_img, pt1=(xx1, yy1), pt2=(xx2, yy2), thickness=7, color=color,
                                  lineType=cv2.LINE_AA)
                    Final_Text += x
        return Final_Text
        # print('한글 -> ' + Final_Text)

    def get_max_temperature(self, thermal_np, x1, y1, x2, y2):
        # 온도 데이터에서 얼굴 영역만 잘라서 검사함
        crop = thermal_np[y1:y2, x1:x2]
        if crop.size == 0:
            return None

        # 얼굴 영역에서 가장 높은 온도 리턴
        return np.max(crop)

    # view camera
    def viewCam(self):
        ret, img = self.cap.read()
        thermal_image_data = thermal_camera.grab()
        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        # 프레임
        # curTime = time.time()
        # sec = curTime - self.prevTime
        # self.prevTime = curTime
        # fps = 1/(sec)
        # FPS = "FPS : %0.1f" % fps
        # cv2.putText(img, FPS, (0, 200), cv2.FONT_HERSHEY_SIMPLEX, 8, (0, 255, 0), thickness=5)
        # read image in BGR format

        h, w = img.shape[:2]

        blob = cv2.dnn.blobFromImage(img, scalefactor=1., size=(300, 300), mean=(104., 177., 123.))
        # Preprocessing. OpenCV의 FaceNet에서 학습시킨대로 Param값을 넣어줌. DNN이 사용하는 형태로 이미지 변환
        # cv2.dnn.blobFromImage함수가 하는 일은 1. Mean subtraction (평균 빼기) / 2.Scaling (이미지 사이즈 바꾸기) / 3.And optionally channel swapping (옵션, 이미지 채널 바꾸기)
        # (104.0,177.0, 123.0)는 mean subtraction의 경험적 최적값. 그럼 mean subtraction이란 RGB값의 일부를 제외해서 dnn이 분석하기 쉽게 단순화해주는 것.
        # (300,300) : dnn모듈이 CNN으로 처리하기 좋은 이미지 사이즈, 모델이 300,300으로 고정

        facenet.setInput(blob)  # 변환해준 이미지 FaceNet의 input
        dets = facenet.forward()  # facedection 결과 저장
        result_img = img.copy()

        # detect face 한뒤, 그 얼굴영역이 마스크 썼을 확률을 계산하여 추가한다.
        for i in range(dets.shape[
                           2]):  # 저장이 된 것을 loop을 돌면서 저장. detections.shape[2]는 모델이 가져오는 최대 박스의 갯수. 200이므로 최대 200개의 얼굴을 인식할수 있다.
            confidence = dets[0, 0, i, 2]
            # 검사하는데 detection의 결과가 자신있는 정도.
            # detections[0, 0]은 우리가 그릴 박스"들"의 속성
            # 따라서 i는 현재 i번째 박스. 2는 세번째 속성이 의미하는데 이게 얼굴일 확률을 나타냄.
            if confidence < 0.5:
                continue

            x1 = int(dets[0, 0, i, 3] * w)  # bounding 박스 구해주기
            y1 = int(dets[0, 0, i, 4] * h)
            x2 = int(dets[0, 0, i, 5] * w)
            y2 = int(dets[0, 0, i, 6] * h)
            # print(i, confidence, x1, y1, x2, y2) i는 몇번째 얼굴인지, cofidence는 실제 얼굴이맞을 확률. 그 뒤는 좌표
            face = img[y1:y2, x1:x2]  # bounding Box을 통해 얼굴만 저장

            # 마스크를 썼나 안썼나 예측
            # 전처리하는 부분
            face_input = cv2.resize(face, dsize=(224, 224))  # 이미지 크기 변경
            face_input = cv2.cvtColor(face_input, cv2.COLOR_BGR2RGB)  # 이미지의 컬러시스템 변경
            face_input = preprocess_input(face_input)  # mobileNetV2에서 하는 preprocessing과 똑같이 하기위해 처리
            face_input = np.expand_dims(face_input,
                                        axis=0)  # 이렇게 하면 shape이 (224,224,3) 으로 나오는데 넣을때는 (1,224,224,3)이 되어야 하므로 차원하나 추가

            mask, self.nomask = model.predict(face_input).squeeze()  # load해놓은 모델에 predict method를 통해, 마스크 여부 확률을 반환

            if mask > self.nomask:
                color = (0, 255, 0)
                label = 'Mask %d%%' % (mask * 100)
            else:
                color = (0, 0, 255)
                label = 'No Mask %d%%' % (self.nomask * 100)

            # mask 썼을확률 계산후 그에대한 결과를 보여주는 곳. 해당 얼굴영역보다 이전 인덱스는 이미 계산되어 이미지에 저장되어 있다.
            cv2.rectangle(result_img, pt1=(x1, y1), pt2=(x2, y2), thickness=7, color=color, lineType=cv2.LINE_AA)
            # 계산된 결과를 현재 돌아가고 있는 얼굴영역 위에 Text를 써줌으로써 표시한다. 마스크 썼을확률은 label에 들어있음.
            cv2.putText(result_img, text=label, org=(x1, y1 - 10), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=4,
                        color=color, thickness=6, lineType=cv2.LINE_AA)
            # 여기서 열 감지
            thermal_np = fir.process_image(thermal_image_data)
            max_temperature = self.get_max_temperature(thermal_np, x1, y1, x2, y2)

            # 마스크 안썻을 확률이 일정확률 이상인 경우
            if self.nomask >= 0.75:
                # 해당 인원 사진 저장
                self.dq.appendleft(face)
                if len(self.dq) == 4:
                    self.dq.pop()
                self.number += 1
                saved_file = 'No_Mask_File/' + str(i) + '_' + str(
                    'No_Mask%d%%_' % (self.nomask * 100) + str(self.number)) + '.jpg'
                cv2.imwrite(saved_file, result_img)
                self.put_img_to_labels(self.dq)
                self.temperature = max_temperature

                IMAGE_FILE = 'No_Mask_File/' + str(i) + '_' + str(
                    'No_Mask%d%%_' % (self.nomask * 100) + str(self.number)) + '.jpg'

                self.Final_Text = self.find_name_and_display(IMAGE_FILE, x1, x2, result_img, color)


                # 전달할 메시지 내용 JSON형식으로 저장후 전달
                message_description = '이름 :' + self.Final_Text + '\n해당인원 온도 :' + str(
                    self.temperature) + '\n마스크 미착용 확률 : ' + str('%d%%' % (self.nomask * 100))

                # telegram 사진 문자 보내는 코드
                f = open(IMAGE_FILE,'rb')
                response = bot.sendPhoto(mc, f)
                response = bot.sendMessage(mc,message_description)

        #
        message_description2 = '이름 :' + self.Final_Text + ' 해당인원 온도 :' + str(self.temperature) + ' 마스크 미착용 확률 : ' + str(
            '%d%%' % (self.nomask * 100))
        self.textdq.appendleft(message_description2)

        self.label_2_text(self.textdq)

        # self.ui.label_2.setText(self.textdq[0])
        # image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        image = cv2.resize(result_img, dsize=(0, 0), fx=0.2, fy=0.2, interpolation=cv2.INTER_LINEAR)
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

if __name__ == '__main__':
    app = QApplication(sys.argv)

    # create and show mainWindow
    mainWindow = MainWindow()
    mainWindow.show()

    sys.exit(app.exec_())
