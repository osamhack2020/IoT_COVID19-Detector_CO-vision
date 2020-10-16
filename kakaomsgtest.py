import json
import requests

with open("kakao_code.json", "r") as fp:
    tokens = json.load(fp)
send_me_url = 'https://kapi.kakao.com/v2/api/talk/memo/default/send'
headers = {
    "Authorization": 'Bearer ' + tokens['access_token']
}
message_description = "안녕하세용"
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
    }
}
data = {
    # 허동준 UUID : MAIwCT4JPggkFiAVJhIhFCMbNwM6CzsLPnY
    # 조동현 UUID : MAIzAjYFNQcxHSgaLh8qHi4aNgI7CjoKP28
    # 친구목록에서 얻어온 UUID 값으로 해야 하므로 수정 필요
    'receiver_uuids': '["MAIzAjYFNQcxHSgaLh8qHi4aNgI7CjoKP28"]',
    "template_object": json.dumps(template)
}
response = requests.post(send_me_url, headers=headers, data=data)
print(response.status_code)
