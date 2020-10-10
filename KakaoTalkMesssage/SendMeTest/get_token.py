import requests
import json

#access Token의 유효시간은 24시간 정도 이므로, 이후에는 Refresh Token을 이용해 재발급 받아야 한다
url = "https://kauth.kakao.com/oauth/token"

data = {
    "grant_type": "authorization_code",
    "client_id": "REST API", #REST API
    "redirect_uri": "https://localhost.com",
    "code": "code" #한번 사용한 코드값은 재사용 불가

}
response = requests.post(url, data=data)
tokens = response.json()
print(tokens)

with open("kakao_token.json", "w") as fp: #토큰값 JSON 파일로 저장
    json.dump(tokens, fp)