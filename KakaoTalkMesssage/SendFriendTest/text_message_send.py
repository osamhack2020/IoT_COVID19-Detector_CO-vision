import json
import requests

get_friend_list_url = 'https://kapi.kakao.com/v1/api/talk/friends'
send_friend_url = 'https://kapi.kakao.com/v1/api/talk/friends/message/default/send'
# 사용자 토큰
token = 'REST API' #REST API
accessToken = 'accessToken'

def getFriendList(accessToken) :
    get_friend_list_url = 'https://kapi.kakao.com/v1/api/talk/friends'
    payload = ''
    headers = {
        'Content-Type' : "application/x-www-form-urlencoded",
        'Cache-Control' : "no-cache",
        'Authorization' : "Bearer " + str(accessToken),
    }
    response = requests.request("GET",get_friend_list_url,data=payload, headers=headers)
    print(response)
    friend_List = json.loads(((response.text).encode('utf-8')))
    friend_UUID_List = []
    elements = response.json().get('elements')
    for element in elements:
        #print(element.get("uuid"))
        friend_UUID_List.append(element.get("uuid"))
    print(friend_UUID_List)
    return friend_UUID_List[0]

print(getFriendList(accessToken))

headers = {
    'Authorization' : "Bearer " + str(accessToken)
}

data = {
  'receiver_uuids': '[""]', #친구 UID 입력
  'template_object': '{ "object_type": "text", "text": "텍스트 영역", "link": { "web_url": "https://developers.kakao.com", "mobile_web_url": "https://developers.kakao.com" }, "button_title": "타이틀 영역" }'
}

response = requests.post(send_friend_url, data=data, headers=headers)
print(response.status_code)
if response.json().get('result_code') == 0:
    print('메시지를 성공적으로 보냈습니다.')
else:
    print('메시지를 성공적으로 보내지 못했습니다. 오류메시지 : ' + str(response.json()))