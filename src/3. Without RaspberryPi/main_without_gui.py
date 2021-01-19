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

# telegram API bot = co_vision_bot
token = 'token-value'
mc = 'value'
bot = telepot.Bot(token)
# 구글 비전 API 설정 (환경변수 설정 / json파일은 서비스 계정 키가 포함된 파일)
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = r'ServiceAccountToken.json'
client = vision.ImageAnnotatorClient() # 사용할 클라이언트 설정

facenet = cv2.dnn.readNet('../training custom dataset/face_detector/deploy.prototxt', '../training custom dataset/face_detector/res10_300x300_ssd_iter_140000.caffemodel')
# FaceDetector 모델 > OpenCv의 DNN
model = load_model('../training custom dataset/mask_detector.model')
# MaskDetector 모델 > Keras 모델

thermal_camera = Lepton()
fir = flir_image_extractor.FlirImageExtractor()

cap = cv2.VideoCapture(0)
# 동영상 로드
ret, img = cap.read()
# ret이 True이면 영상이 있다는 뜻
fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
out = cv2.VideoWriter('output.mp4', fourcc, 1, (img.shape[1], img.shape[0]))
# cv2.VideoWriter(outputFile, fourcc, frame, size) : fourcc는 코덱 정보, frame은 초당 저장될 프레임, size는 저장될 사이즈를 뜻한다. cv2.VideoWriter_fourcc('D','I','V','X') 이런식으로 사용
number = 0  # 마스크 안 쓴 사람 사진 저장할 때 사용

def get_max_temperature(thermal_np, x1, y1, x2, y2):
    # 온도 데이터에서 얼굴 영역만 잘라서 검사함
    crop = thermal_np[y1:y2, x1:x2]
    if crop.size == 0:
        return None

    # 얼굴 영역에서 가장 높은 온도 리턴
    return np.max(crop)

# google vision API
def find_name_and_display(IMAGE_FILE,x1,x2,result_img,color):
    with io.open(IMAGE_FILE, 'rb') as image_file:
        content = image_file.read()
    image = vision.Image(content=content) #이미지 파일 넘겨줌

    response = client.document_text_detection(image=image)
    #군복 이름표의 경우 주변환경에 의해 항상 같은 모습으로 촬영되지 않으므로 필기 입력 감지 이용
    #response에는 상세 정보들이 저장. 어느 언어로 인식 했는지 부터 문장 별, 단어 별, 각 철자 별 어떻게 인식을 하였는지, 이미지에서 위치는 어디에 있는지 등의 정보가 담김.
    #response의 text_annotations에는 내용을 간추려 철자를 제외한 문장과 단어에 대한 정보를 담음

    Final_Text = "" #읽은 이름을 저장할 변수
    for data in response.text_annotations:
        xx1 = data.bounding_poly.vertices[0].x - 60  # 표시될 사각형이 너무 오른쪽으로 튀어나와 좌표 수정
        yy1 = data.bounding_poly.vertices[0].y
        xx2 = data.bounding_poly.vertices[2].x
        yy2 = data.bounding_poly.vertices[2].y + 20
        if xx1 > (x1 + x2) // 2 or xx2 > (x1 + x2) // 2:  # 이름표가 오른쪽 가슴에 있으므로 얼굴 왼쪽은 무시함
            continue
        for x in data.description:
            if ord('가') <= ord(x) <= ord('힣'): # 한글이외의 글자들은 모두 걸러내는 텍스트 가공과정
                # cv2.rectangle(result_img, pt1=(xx1, yy1), pt2=(xx2, yy2), thickness=7, color=color,
                #              lineType=cv2.LINE_AA)
                Final_Text += x
    return Final_Text
while cap.isOpened():
    ret, img = cap.read()
    thermal_image_data = thermal_camera.grab()
    if not ret:
        break
    # Optional step 영상이 돌려져 있으면 돌리기
    img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
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

        mask, nomask = model.predict(face_input).squeeze()  # load해놓은 모델에 predict method를 통해, 마스크 여부 확률을 반환

        if mask > nomask:
            color = (0, 255, 0)
            label = 'Mask %d%%' % (mask * 100)
        else:
            color = (0, 0, 255)
            label = 'No Mask %d%%' % (nomask * 100)

        # mask 썼을확률 계산후 그에대한 결과를 보여주는 곳. 해당 얼굴영역보다 이전 인덱스는 이미 계산되어 이미지에 저장되어 있다.
        cv2.rectangle(result_img, pt1=(x1, y1), pt2=(x2, y2), thickness=2, color=color, lineType=cv2.LINE_AA)
        # 계산된 결과를 현재 돌아가고 있는 얼굴영역 위에 Text를 써줌으로써 표시한다. 마스크 썼을확률은 label에 들어있음.
        cv2.putText(result_img, text=label, org=(x1, y1 - 10), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.8, color=color, thickness=2, lineType=cv2.LINE_AA)

        # 여기서 열 감지
        thermal_np = fir.process_image(thermal_image_data)
        max_temperature = get_max_temperature(thermal_np, x1, y1, x2, y2)

        # 마스크 안썻을 확률이 일정확률 이상인 경우 + 얼굴 온도가 37.5도 이상인 경우
        if nomask >= 0.75 or max_temperature >= 37.5:
            # 해당 인원 사진 저장
            number += 1
            temperature = max_temperature
            cv2.imwrite('No_Mask-High_Temp/' + str(i)+'_'+str('No_Mask%d%%_' % (nomask * 100) + str(number)) + 'Temp_' + str(max_temperature) + '.jpg', result_img)
            IMAGE_FILE = 'No_Mask-High_Temp/' + str(i)+'_'+str('No_Mask%d%%_' % (nomask * 100) + str(number)) + 'Temp_' + str(max_temperature) + '.jpg'

            with io.open(IMAGE_FILE, 'rb') as image_file:
                content = image_file.read()
            image = vision.Image(content=content)
            response = client.document_text_detection(image=image)
            Final_Text = find_name_and_display(IMAGE_FILE, x1, x2, result_img, color)

            message_description = '이름 :' + Final_Text + '\n해당인원 온도 :' + str(temperature) + '\n마스크 미착용 확률 : ' + str('%d%%' % (nomask * 100))
            f = open(IMAGE_FILE,'rb')
            response = bot.sendPhoto(mc, f)
            response = bot.sendMessage(mc,message_description)

    out.write(result_img)
    resized_img = cv2.resize(result_img, dsize=(0, 0), fx=0.3, fy=0.3, interpolation=cv2.INTER_LINEAR)
    cv2.imshow('result', resized_img)  # 실시간 모니터링하고 있는 화면을 띄워줌
    if cv2.waitKey(1) == ord('q'):  # q누르면 동영상 종료
        break
out.release()
cap.release()
