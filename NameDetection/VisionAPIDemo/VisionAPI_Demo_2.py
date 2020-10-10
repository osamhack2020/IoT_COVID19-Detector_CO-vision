#this is for handwriting images but I think it works better than VisionAPI_Demo.py

import os, io
from google.cloud import vision
#import pandas as pd

os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = r'ServiceAccountToken.json'
client = vision.ImageAnnotatorClient()

IMAGE_FILE = 'jun_name1.png'
FOLDER_PATH = r'C:\Users\Administrator\anaconda3\envs\VisionAPIDemo'
FILE_PATH = os.path.join(FOLDER_PATH, IMAGE_FILE)

with io.open(FILE_PATH, 'rb') as image_file:
    content = image_file.read()

image = vision.Image(content=content)
response = client.document_text_detection(image=image)
docText = response.full_text_annotation.text
#print(docText)

Final_Text = ""
Flag = False
for x in docText:
    if ord('가') <= ord(x) <= ord('힣'):
        Flag = True
        Final_Text += x
if Flag:
    print('한국어 감지 성공 ->' + Final_Text)