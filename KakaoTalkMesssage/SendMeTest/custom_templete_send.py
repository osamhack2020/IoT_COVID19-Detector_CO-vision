#test 20.10.09
import requests

# 커스텀 템플릿 주소 : https://kapi.kakao.com/v2/api/talk/memo/send
talk_url = "https://kapi.kakao.com/v2/api/talk/memo/send"

# 사용자 토큰
token = 'REST API' #REST API
accessToken = 'accessToken'
header = {
    "Authorization": 'Bearer ' + str(accessToken).format(
        token=token
    )
}

# 메시지 template id와 정의했던 ${Image}을 JSON 형식으로 값으로 입력
payload = {
    'template_id' : '38324',
    #이미지 링크, 온도, 마스크 착용 퍼센테이지 전달
    'template_args' : '{"Image": "IMAGE_URL, 클라이언트의 사진을 가져오거나 서버의 사진을 가져오기가 아닌 URL상에서 가져와야함", "Temperature": "읽어온 온도 변수 입력", "Mask": "읽어온 마스크 착용확률 퍼센테이지 입력"}'
}

# 카카오톡 메시지 전송
res = requests.post(talk_url, data=payload, headers=header)

if res.json().get('result_code') == 0:
    print('메시지를 성공적으로 보냈습니다.')
else:
    print('메시지를 성공적으로 보내지 못했습니다. 오류메시지 : ' + str(res.json()))