#https://www.youtube.com/watch?v=xKvffLRSyPk&list=PL3JVwFmb_BnSLFyVThMfEavAEZYHBpWEd&index=1

import os, io
from google.cloud import vision
import pandas as pd
#from google.cloud.vision import types -> 버전 업그레이드 되면서 types 사용 안함

os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = r'ServiceAccountToken.json'

client = vision.ImageAnnotatorClient()

FILE_NAME = 'jun_name1.png'
FOLDER_PATH = r'C:\Users\Administrator\anaconda3\envs\VisionAPIDemo'

with io.open(os.path.join(FOLDER_PATH, FILE_NAME), 'rb') as image_file:
    content = image_file.read()
image = vision.Image(content=content)

response = client.text_detection(image=image)

df = pd.DataFrame(columns=['locale', 'description'])

texts = response.text_annotations
for text in texts:
    df = df.append(
        dict(
            locale=text.locale,
            description=text.description
        ),
        ignore_index=True
    )

print(df['description'][0])
print('finish')
