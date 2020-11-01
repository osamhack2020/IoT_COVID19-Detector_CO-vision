# Data Set을 만들어서 학습.
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import load_model
import numpy as np
# import matplotlib.pyplot as plt
import os, io
import json
import requests
from google.cloud import vision
#########################################################################텔레그렘 수정 내용
import telepot

# import some PyQt5 modules
from PyQt5.QtWidgets import QApplication
from PyQt5.QtWidgets import QWidget
from PyQt5.QtGui import QImage
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import QTimer

# import Opencv module
import cv2

from desktop.ui_main_window_test02 import *

from collections import deque

# bot = co_vision_bot
token = '1130712531:AAE3W0J9Y3s2opGvE_c8My8e96-vhqlLAGE'
mc = '1314303321'
bot = telepot.Bot(token)
####################################### ##################################
# 구글 API 설정
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = r'ServiceAccountToken.json'
client = vision.ImageAnnotatorClient()

# 카카오톡 메시지 커스텀 템플릿 주소 : https://kapi.kakao.com/v2/api/talk/memo/send
talk_url = "https://kapi.kakao.com/v2/api/talk/memo/send"
get_friend_list_url = 'https://kapi.kakao.com/v1/api/talk/friends'
send_me_url = 'https://kapi.kakao.com/v2/api/talk/memo/default/send'
send_friend_url = 'https://kapi.kakao.com/v1/api/talk/friends/message/default/send'
# 카카오 사용자 토큰
token = 'REST API'  # REST API
accessToken = 'accessToken'
headers = {
    "Authorization": 'Bearer ' + str(accessToken).format(
        token=token
    )
}


# 카카오 친구 목록 얻어오기. 수정 필요
def getFriendList(accessToken):
    payload = ''
    headers = {
        'Content-Type': "application/x-www-form-urlencoded",
        'Cache-Control': "no-cache",
        'Authorization': "Bearer " + str(accessToken),
    }
    response = requests.request("GET", get_friend_list_url, data=payload, headers=headers)
    # print(response)
    friend_List = json.loads(((response.text).encode('utf-8')))
    friend_UUID_List = []
    elements = response.json().get('elements')
    for element in elements:
        # print(element.get("uuid"))
        friend_UUID_List.append(element.get("uuid"))
    # print(friend_UUID_List)
    return friend_UUID_List[0]


facenet = cv2.dnn.readNet('MaskDetection/models/deploy.prototxt', 'MaskDetection/models/res10_300x300_ssd_iter_140000.caffemodel')
# FaceDetector 모델 > OpenCv의 DNN
model = load_model('MaskDetection/models/mask_detector.model')
# MaskDetector 모델 > Keras 모델
#cap = cv2.VideoCapture('imgs/junha_video.mp4')
# 동영상 로드
# 노트북 캠의 실시간 영상을 받아오고 싶으면 0을 넣으면 된다!
#ret, img = cap.read()
# ret이 True이면 영상이 있다는 뜻
#fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
#out = cv2.VideoWriter('output.mp4', fourcc, 1, (img.shape[1], img.shape[0]))
# cv2.VideoWriter(outputFile, fourcc, frame, size) : fourcc는 코덱 정보, frame은 초당 저장될 프레임, size는 저장될 사이즈를 뜻합니다 cv2.VideoWriter_fourcc('D','I','V','X') 이런식으로 사용
# 현재 테스트 동영상의 프레임은 25
#number = 0  # 마스크 안 쓴 사람 사진 저장할 때 사용
































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
        # self.cap =  cv2.VideoCapture(0) 으로 하면 웹캠 실시간으로 나옴
        # self.cap =  cv2.VideoCapture('imgs/junha_video.mp4')
        self.cap =  cv2.VideoCapture('imgs/junha_video.mp4')
        # self.capthermal = cv2.VideoCapture('imgs/junha_video.mp4')
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
        self.nomask=0
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


    def find_name_and_display(self, IMAGE_FILE,x1,x2,result_img,color):
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
        #print('한글 -> ' + Final_Text)
    def sound(self):
        pass

    # view camera
    def viewCam(self):
        ret, img = self.cap.read()
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
            # print(i, confidence, x1, y1, x2, y2) i는 몇번째 얼굴인지, cofidence는 실제 얼굴이맞을 확률. 그 뒤는 좌표
            face = img[y1:y2, x1:x2]  # bounding Box을 통해 얼굴만 저장

            # 마스크를 썼나 안썼나 예측
            # 전처리하는 부분
            face_input = cv2.resize(face, dsize=(224, 224))  # 이미지 크기 변경
            face_input = cv2.cvtColor(face_input, cv2.COLOR_BGR2RGB)  # 이미지의 컬러시스템 변경
            face_input = preprocess_input(face_input)  # mobileNetV2에서 하는 preprocessing과 똑같이 하기위해 처리
            face_input = np.expand_dims(face_input, axis=0)  # 이렇게 하면 shape이 (224,224,3) 으로 나오는데 넣을때는 (1,224,224,3)이 되어야 하므로 차원하나 추가

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

            # 마스크 안썻을 확률이 일정확률 이상인 경우
            if self.nomask >= 0.75:
                # 해당 인원 사진 저장
                self.dq.appendleft(face)
                if len(self.dq)==4:
                    self.dq.pop()
                self.number += 1
                saved_file = 'No_Mask_File/' + str(i)+'_'+str('No_Mask%d%%_' % (self.nomask * 100) + str(self.number)) + '.jpg'
                cv2.imwrite(saved_file, result_img)
                self.put_img_to_labels(self.dq)
                self.temperature = 36.5  # 현재 온도 변수가 없으므로 임시로 설정


                #############################################################################
                # GoogleVisionAPI branch  에서 추가한 내용
                IMAGE_FILE = 'No_Mask_File/' + str(i) + '_' + str('No_Mask%d%%_' % (self.nomask * 100) + str(self.number)) + '.jpg'
                # FOLDER_PATH = r'C:\Users\Administrator\anaconda3\envs\VisionAPIDemo'
                # FILE_PATH = os.path.join(FOLDER_PATH, IMAGE_FILE)
                # Name_img = img[y2:h, 0:(x1+x2)/2]

                # with io.open(Name_img, 'rb') as image_file:
                #     content = image_file.read()



                """
                name_img = img[y2:h, 0:(x1+x2)//2]
                is_success, im_buf_arr = cv2.imencode(".jpg",name_img)
                io_buf = io.BytesIO(im_buf_arr)
                byte_im = io_buf.getvalue()
                image = vision.Image(content=byte_im)
                response = client.document_text_detection(image=image)
                docText = response.full_text_annotation.text
                """



                # with io.open(IMAGE_FILE, 'rb') as image_file:
                #     content = image_file.read()

                # image = vision.Image(content=content)
                # response = client.document_text_detection(image=image)
                # docText = response.full_text_annotation.text




                    
                self.Final_Text = self.find_name_and_display(IMAGE_FILE,x1,x2,result_img,color)

                


                # with io.open(IMAGE_FILE, 'rb') as image_file:
                #     content = image_file.read()

                # image = vision.Image(content=content)
                # response = client.document_text_detection(image=image)
                # self.Final_Text = ""
                # for data in response.text_annotations:
                #     xx1 = data.bounding_poly.vertices[0].x - 60 # 박스가 너무 오른쪽으로 나옴 그래서 수정함.
                #     yy1 = data.bounding_poly.vertices[0].y
                #     xx2 = data.bounding_poly.vertices[2].x
                #     yy2 = data.bounding_poly.vertices[2].y + 20
                #     if xx1 > (x1+x2)//2 or xx2 > (x1+x2)//2:
                #         continue
                #     for x in data.description:
                #         if ord('가') <= ord(x) <= ord('힣'):
                #             cv2.rectangle(result_img, pt1=(xx1, yy1), pt2=(xx2, yy2), thickness=7, color=color, lineType=cv2.LINE_AA)
                #             self.Final_Text += x

                # print('한글 -> ' + self.Final_Text)
                #self.ui.label_2.setText(Final_Text)

                    # cv2.rectangle(img, pt1=(xx1, yy1), pt2=(xx2, yy2), thickness=2, color=color, lineType=cv2.LINE_AA)
                    # img_resize = cv2.resize(result_img, dsize=(0, 0), fx=0.2, fy=0.2, interpolation=cv2.INTER_LINEAR)
                    # cv2.imshow('fram', img_resize)
                    # cv2.waitKey(0)














                """
                # 한글만 가져오는 코드
                Final_Text = ""
                Flag = False
                for x in docText:
                    if ord('가') <= ord(x) <= ord('힣'):
                        Flag = True
                        Final_Text += x

                print('한글 -> ' + Final_Text)

                #############################################################################
                message_description = '이름 :' + Final_Text + '\n해당인원 온도 :' + str(temperature) + '\n마스크 미착용 확률 : ' + str('%d%%' % (nomask * 100))
                
                
                """

                # 전달할 메시지 내용 JSON형식으로 저장후 전달
                message_description = '이름 :' + self.Final_Text + '\n해당인원 온도 :' + str(self.temperature) + '\n마스크 미착용 확률 : ' + str('%d%%' % (self.nomask * 100))
                # template = {
                #     "object_type": "feed",
                #     "content": {
                #         "image_url": "IMAGE_URL, 클라이언트의 사진을 가져오거나 서버의 사진을 가져오기가 아닌 URL상에서 가져와야함",
                #         "title": "이상증상자 및 마스크 미착용자 식별",
                #         "description": message_description,
                #         "image_width": 640,
                #         "image_height": 640,
                #         "link": {
                #             "web_url": "http://www.daum.net",
                #             "mobile_web_url": "http://m.daum.net",
                #         }
                #     }
                # }
                # data = {
                #     # 허동준 UUID : MAIwCT4JPggkFiAVJhIhFCMbNwM6CzsLPnY
                #     # 조동현 UUID : MAIzAjYFNQcxHSgaLh8qHi4aNgI7CjoKP28
                #     # 친구목록에서 얻어온 UUID 값으로 해야 하므로 수정 필요
                #     'receiver_uuids': '["MAIzAjYFNQcxHSgaLh8qHi4aNgI7CjoKP28"]',
                #     "template_object": json.dumps(template)
                # }
                # # 메시지 전송 및 오류 검출
                # response = requests.post(send_friend_url, headers=headers, data=data)
                # print(response.status_code)
                # if response.json().get('result_code') == 0:
                #     print('메시지를 성공적으로 보냈습니다.')
                # else:
                #     print('메시지를 성공적으로 보내지 못했습니다. 오류메시지 : ' + str(response.json()))

        #out.write(result_img)
        #resized_img = cv2.resize(result_img, dsize=(0, 0), fx=0.3, fy=0.3, interpolation=cv2.INTER_LINEAR)
        #cv2.imshow('result', result_img)  # 실시간 모니터링하고 있는 화면을 띄워줌


















        # # telegram 사진 문자 보내는 코드
        # f = open(IMAGE_FILE,'rb')
        # response = bot.sendPhoto(mc, f)
        # response = bot.sendMessage(mc,message_description)


        #
        message_description2 = '이름 :' + self.Final_Text + ' 해당인원 온도 :' + str(self.temperature) + ' 마스크 미착용 확률 : ' + str('%d%%' % (self.nomask * 100))
        self.textdq.appendleft(message_description2)
    
        self.label_2_text(self.textdq)

        

        #self.ui.label_2.setText(self.textdq[0])
        #image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
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
