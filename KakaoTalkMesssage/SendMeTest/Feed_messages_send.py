import json
import requests

headers = {
    "Authorization": "Bearer " + 'accessToken',
}

temperature = 36.5
Mask = 76
message_description = '해당인원 온도 :' +str(temperature) + '\n마스크 미착용 확률 : ' + str(Mask) +'%'

template = {
    "object_type": "feed",
    "content": {
            "image_url": "IMAGE_URL, 클라이언트의 사진을 가져오거나 서버의 사진을 가져오기가 아닌 URL상에서 가져와야함",
            "title": "이상증상자 및 마스크 미착용자 식별",
            "description": message_description,
            "image_width": 640,
            "image_height": 640,
            "link": {
            "web_url": "http://www.daum.net",
            "mobile_web_url": "http://m.daum.net",
            }
        },
    "buttons": [
            {
            "title": "웹으로 이동",
            "link": {
                "web_url": "http://www.daum.net",
                "mobile_web_url": "http://m.daum.net"
            }
            },

            {
            "title": "", #APP으로 이동 삭제
            "link": {
                "android_execution_params": "contentId=100",
                "ios_execution_params": "contentId=100"
            }
            }

        ]
    }

data = {
    "template_object": json.dumps(template)
}

response = requests.post('https://kapi.kakao.com/v2/api/talk/memo/default/send', headers=headers, data=data)
print(response.status_code)
if response.json().get('result_code') == 0:
    print('메시지를 성공적으로 보냈습니다.')
else:
    print('메시지를 성공적으로 보내지 못했습니다. 오류메시지 : ' + str(response.json()))