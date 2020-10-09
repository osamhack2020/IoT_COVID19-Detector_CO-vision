import requests
# 사용자 토큰
token = 'REST API' #REST API
accessToken = 'accessToken'

headers = {
    'Authorization' : "Bearer " + str(accessToken)
}

data = {
  'receiver_uuids': '["abcdefg0001"]', #친구 UID 입력
  'template_object': '{ "object_type": "text", "text": "텍스트 영역", "link": { "web_url": "https://developers.kakao.com", "mobile_web_url": "https://developers.kakao.com" }, "button_title": "타이틀 영역" }'
}

response = requests.post('https://kapi.kakao.com/v1/api/talk/friends/message/default/send', headers=headers, data=data)
