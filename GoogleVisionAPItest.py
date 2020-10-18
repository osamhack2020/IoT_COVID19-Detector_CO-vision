import numpy as np
import cv2
# import matplotlib.pyplot as plt
import os, io
import json
import requests
from google.cloud import vision
#########################################################################텔레그렘 수정 내용
import telepot
# 구글 API 설정
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = r'ServiceAccountToken.json'
client = vision.ImageAnnotatorClient()

# cap = cv2.VideoCapture('imgs/junha_video.mp4')
# # 동영상 로드
# # 노트북 캠의 실시간 영상을 받아오고 싶으면 0을 넣으면 된다!

# ret, img = cap.read()
IMAGE_FILE = 'NO_Mask_File/0_No_Mask99%_1.jpg'


with io.open(IMAGE_FILE, 'rb') as image_file:
    content = image_file.read()

image = vision.Image(content=content)
response = client.document_text_detection(image=image)
texts = response.text_annotations
img = cv2.imread(IMAGE_FILE)
color = (0, 0, 255)
for data in response.text_annotations:
    x1 = data.bounding_poly.vertices[0].x
    y1 = data.bounding_poly.vertices[0].y
    x2 = data.bounding_poly.vertices[2].x
    y2 = data.bounding_poly.vertices[2].y

    cv2.rectangle(img, pt1=(x1, y1), pt2=(x2, y2), thickness=2, color=color, lineType=cv2.LINE_AA)
    img_resize = cv2.resize(img, dsize=(0, 0), fx=0.2, fy=0.2, interpolation=cv2.INTER_LINEAR)
    cv2.imshow('fram', img_resize)
    cv2.waitKey(0)


        


    
    
    
#docText = response.full_text_annotation.text