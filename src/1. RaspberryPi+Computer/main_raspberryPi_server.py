# 라즈베리 파이와 연결된 컴퓨터나 노트북에서 실행할 코드
import imagezmq
import flir_image_extractor
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import os
import io
from google.cloud import vision
import telepot

# import some PyQt5 modules
from PyQt5.QtWidgets import QApplication
from PyQt5.QtWidgets import QWidget
from PyQt5.QtGui import QImage
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import QTimer

from ui_main_window import *

from collections import deque

# telegram API bot = co_vision_bot
token = 'telegram-token-value'
mc = 'value'
bot = telepot.Bot(token)
# 구글 비전 API 설정 (환경변수 설정 / json파일은 서비스 계정 키가 포함된 파일)
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = r'ServiceAccountToken.json'
client = vision.ImageAnnotatorClient() # 사용할 클라이언트 설정

facenet = cv2.dnn.readNet('../training custom dataset/face_detector/deploy.prototxt', '../training custom dataset/face_detector/res10_300x300_ssd_iter_140000.caffemodel')
# FaceDetector 모델 > OpenCv의 DNN
model = load_model('../training custom dataset/mask_detector.model')
# MaskDetector 모델 > Keras 모델

fir = flir_image_extractor.FlirImageExtractor()

class MainWindow(QWidget):
    # class constructor
    def __init__(self):
        # call QWidget constructor
        super().__init__()
        self.ui = Ui_Form()
        self.ui.setupUi(self)

        #라즈베리파이로부터 이미지를 전송받기 위해 기초 변수 설정
        self.image_hub = imagezmq.ImageHub()
        self.name, self.temp_img = self.image_hub.recv_image()
        rpi, self.visible_img = self.image_hub.recv_image()  # 가시광선 카메라 현재화면 이미지 read
        rpi, self.thermal_image_data = self.image_hub.recv_image()  # 적외선 카메라 현재화면을 이미지로 grab
        self.image_hub.send_reply(b'OK')  # 요청-응답 패턴을 사용하므로 수신 성공메시지를 보내야 함

        # create a timer
        self.timer = QTimer()
        self.timer_recv_img = QTimer()
        # 일정시간 마다 반복할 작업 목록
        self.timer_recv_img.timeout.connect(self.recv_img)
        self.timer.timeout.connect(self.viewCam)
        self.timer.timeout.connect(self.viewThermalCam)

        self.number = 0 # 마스크 안 쓴 사람 사진 저장할 때 사용
        self.dq = deque()
        self.textdq = deque([])

        # start timer
        self.timer_recv_img.start(1) #가시광선 이미지와 적외선 이미지가 번갈아서 오기때문에 시간을 짧게 잡고가야함
        self.timer.start(20) #20/1000초로 반복
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
        self.response = 0

    def recv_img(self):
        self.name, self.temp_img = self.image_hub.recv_image()  # 이미지를 지속적으로 전송하고 계속해서 전송받으므로 가시광선 이미지와 적외선 이미지가 교차하여 들어오는 것을 분류
        if self.name == 'visible_image':
            rpi, self.visible_img = self.image_hub.recv_image()  # 가시광선 카메라 현재화면 이미지 read
        elif self.name == 'thermal_image':
            rpi, self.thermal_image_data = self.image_hub.recv_image()  # 적외선 카메라 현재화면을 이미지로 grab
        self.image_hub.send_reply(b'OK')  # 요청-응답 패턴을 사용하므로 지속적으로 수신 성공메시지를 보내야 함
        #self.image_hub.close()

    def put_img_to_labels(self, dq): #라벨에 이미지를 보여주는 함수. 현재는 특이 인원 감지시 라벨에 그 캡처화면을 띄우는 용도로 사용
        image = self.dq[0] #가장 최근 이미지를 저장
        image = cv2.resize(image, dsize=(0, 0), fx=0.2, fy=0.2, interpolation=cv2.INTER_LINEAR) #라벨에 맞게 넣어야되므로 이미지를 비율에 맞게 리사이즈
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width, channel = image.shape
        step = channel * width
        qImg = QImage(image.data, width, height, step, QImage.Format_RGB888)
        self.ui.label.setPixmap(QPixmap.fromImage(qImg)) #라벨에 가장 최근이미지 설정

    def label_2_text(self, textdq): #로그 띄우는 용도로 사용중
        TEXT = ""
        if len(self.textdq) > 10: #현재 덱에있는 상세정보의 개수가 10개 이상이면 가장 오래된 오른쪽부터 pop
            textdq.pop()
        for text in textdq: #덱에 있는 상세정보를 하나씩 TEXT에 줄바꿈으로 구분하여 저장
            TEXT += text + '\n'
        self.ui.label_2.setText(TEXT) #label_2 라벨에 텍스트 내용은 TEXT에 있는 내용으로 저장해서 보여줌

    def find_name_and_display(self, IMAGE_FILE, x1, x2, result_img, color):
        with io.open(IMAGE_FILE, 'rb') as image_file:
            content = image_file.read()

        image = vision.Image(content=content) #이미지 파일 넘겨줌
        self.response = client.document_text_detection(image=image)
        #군복 이름표의 경우 주변환경에 의해 항상 같은 모습으로 촬영되지 않으므로 필기 입력 감지 이용
        #response에는 상세 정보들이 저장. 어느 언어로 인식 했는지 부터 문장 별, 단어 별, 각 철자 별 어떻게 인식을 하였는지, 이미지에서 위치는 어디에 있는지 등의 정보가 담김.
        #response의 text_annotations에는 내용을 간추려 철자를 제외한 문장과 단어에 대한 정보를 담음

        Final_Text = "" # 읽은 이름을 저장할 변수
        for data in self.response.text_annotations:
            xx1 = data.bounding_poly.vertices[0].x - 60  # 표시될 사각형이 너무 오른쪽으로 튀어나와 좌표 수정
            yy1 = data.bounding_poly.vertices[0].y
            xx2 = data.bounding_poly.vertices[2].x
            yy2 = data.bounding_poly.vertices[2].y + 20

            if xx1 > (x1 + x2) // 2 or xx2 > (x1 + x2) // 2: # 이름표가 오른쪽 가슴에 있으므로 얼굴 왼쪽은 무시하도록 얼굴 절반을 기준으로 우측 이미지만 이미지 검출 대상으로 만듦
                continue

            for x in data.description: # 한글이외의 글자들은 모두 걸러내는 텍스트 가공과정
                if ord('가') <= ord(x) <= ord('힣'):
                    cv2.rectangle(result_img, pt1=(xx1, yy1), pt2=(xx2, yy2), thickness=7, color=color, lineType=cv2.LINE_AA)
                    Final_Text += x
        return Final_Text
        # print('한글 -> ' + Final_Text)

    def get_max_temperature(self, thermal_np, x1, y1, x2, y2):
        # 온도 데이터에서 얼굴 영역만 잘라서 검사함
        crop = thermal_np[y1:y2, x1:x2]
        if crop.size == 0: #얼굴이 검출이 안되면 crop의 size는 0
            return None

        # 얼굴 영역에서 가장 높은 온도 리턴
        return np.max(crop)

    # view camera
    def viewCam(self):
        img = self.visible_img #가시광선 카메라 현재화면을 이미지로 read
        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE) #가시광선 카메라 모습을 정상적으로 읽기위해 rotate. 카메라가 비추는 방향에 따라 삭제또는 유지

        h, w = img.shape[:2]

        blob = cv2.dnn.blobFromImage(img, scalefactor=1., size=(300, 300), mean=(104., 177., 123.))
        # Preprocessing. OpenCV의 FaceNet에서 학습시킨대로 Param값을 넣어줌. DNN이 사용하는 형태로 이미지 변환
        # cv2.dnn.blobFromImage함수가 하는 일은 1. Mean subtraction (평균 빼기) / 2.Scaling (이미지 사이즈 바꾸기) / 3.And optionally channel swapping (옵션, 이미지 채널 바꾸기)
        # (104.0,177.0, 123.0)는 mean subtraction의 경험적 최적값. 그럼 mean subtraction이란 RGB값의 일부를 제외해서 dnn이 분석하기 쉽게 단순화해주는 것.
        # (300,300) : dnn모듈이 CNN으로 처리하기 좋은 이미지 사이즈, 모델이 300,300으로 고정

        facenet.setInput(blob)  # 변환해준 이미지 FaceNet의 input
        dets = facenet.forward()  # facedection 결과 저장
        result_img = img.copy()

        # detect face 한뒤, 그 얼굴영역이 마스크 썼을 확률을 계산하여 추가한다. 얼굴 인식한 값을 이용해 이미지 처리를 진행하는 부분
        for i in range(dets.shape[2]):  # 저장이 된 것을 loop을 돌면서 저장. detections.shape[2]는 모델이 가져오는 최대 박스의 갯수. 200이므로 최대 200개의 얼굴을 인식할수 있다.
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
            # print(i, confidence, x1, y1, x2, y2) i는 몇번째 얼굴인지, cofidence는 실제 얼굴이 맞을 확률. 그 뒤는 좌표
            face = img[y1:y2, x1:x2]  # bounding Box을 통해 얼굴 이미지만 저장


            # 마스크 착용여부 체크 코드
            # 전처리하는 부분
            face_input = cv2.resize(face, dsize=(224, 224))  # 이미지 크기 변경
            face_input = cv2.cvtColor(face_input, cv2.COLOR_BGR2RGB)  # 이미지의 컬러시스템 변경
            face_input = preprocess_input(face_input)  # mobileNetV2에서 하는 preprocessing과 똑같이 하기위해 처리
            face_input = np.expand_dims(face_input,axis=0)  # 이렇게 하면 shape이 (224,224,3) 으로 나오는데 넣을때는 (1,224,224,3)이 되어야 하므로 차원하나 추가

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
            cv2.putText(result_img, text=label, org=(x1, y1 - 10), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=4, color=color, thickness=6, lineType=cv2.LINE_AA)

            # 체온체크 코드
            thermal_np = fir.process_image(self.thermal_image_data) # 적외선 카메라에서 따온 이미지를 Flir Image Extractor Library을 이용하여 이미지 처리
            max_temperature = self.get_max_temperature(thermal_np, x1, y1, x2, y2) # 가공된 이미지를 이용해 기존에 구한 얼굴 영역만을 대상으로 가장 높은 온도 반환

            # 마스크 미착용 확률이 일정확률 이상 이거나 얼굴 영역 최고온도가 고열인 경우 해당 인원 사진과 상세정보를 띄워주기 위함
            if self.nomask >= 0.75 or max_temperature >= 37.5:
                # 해당 인원 사진 저장
                self.dq.appendleft(face) #해당 사진 덱에 가장 왼쪽에 저장
                if len(self.dq) == 4: #사진이 4장이되면 가장 오래된 사진 pop
                    self.dq.pop()
                self.number += 1 #사람 순서 저장
                IMAGE_FILE = 'No_Mask-High_Temp/' + str(i) + '_' + str('No_Mask%d%%_' % (self.nomask * 100) + str(self.number)) + 'Temp_' + str(max_temperature) + '.jpg'
                cv2.imwrite(IMAGE_FILE, result_img)

                #이미지 처리를 위해 임시저장하고 임시저장한 이미지 이용하기위해 임시저장한 파일의 이름도 같이 텍스트로 저장

                saved_file = 'No_Mask_File/' + str(i) + '_' + str('No_Mask%d%%_' % (self.nomask * 100) + str(self.number)) + '.jpg'
                cv2.imwrite(saved_file, result_img) #임시가 아닌 최종이미지 저장
                self.put_img_to_labels(self.dq) #캡쳐된 이미지를 라벨에 넣어서 이미지를 보여주기 위해 호출. 해당 호출로 라벨에 이미지 처리되어 특이사항이 감지된 캡처화면이 나오게 된다
                self.Final_Text = self.find_name_and_display(IMAGE_FILE, x1, x2, result_img, color) #이름을 추출하기 위해 호출

                # 전달할 메시지 내용 저장후 전달
                message_description = '이름 :' + self.Final_Text + '\n해당인원 온도 :' + str(max_temperature) + '\n마스크 미착용 확률 : ' + str('%d%%' % (self.nomask * 100))

                # telegram 사진 문자 보내는 코드
                f = open(IMAGE_FILE,'rb')
                self.response = bot.sendPhoto(mc, f)
                self.response = bot.sendMessage(mc,message_description)

        message_description2 = '이름 :' + self.Final_Text + '해당인원 온도 :' + str(self.temperature) + '마스크 미착용 확률 : ' + str('%d%%' % (self.nomask * 100))
        self.textdq.appendleft(message_description2) #특이사항 인원 상세정보를 덱의 가장 첫순서에 삽입

        self.label_2_text(self.textdq) #상세정보가 저장된 덱을 GUI로 보여주기 위해 함수 호출 (로그를 보여줌)

        # 최종적으로 현재 이미지 처리된 영상부분을 띄워주는 부분. result_img를 resize하여 QPixmap을 이용해 main_video로 띄워줌
        # image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE) #이미 가시광선 카메라는 rotate하고나서 이미지처리 했으므로 주석처리
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
        image = self.thermal_image_data
        image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE) #적외선 카메라 모습을 정상적으로 띄우기 위해 rotate. 카메라가 비추는 방향에 따라 삭제또는 유지
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