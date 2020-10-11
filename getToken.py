import requests

app_key = "6f93da19236547496407bf00a5a5cb7e"
code = "6ZU1GCjg84JrehQpcyI4hlRj6nNrBYzcHSaVFUxECKQhdLYmnzwgwNjjVDUieqpoKp7ONgopcBQAAAF1FVPW1Q"

url = "https://kauth.kakao.com/oauth/token"
data = {
    "grant_type" : "authorization_code",
    "client_id" : app_key,
    "redirect_url" : "https://localhost.com",
    "code" : code
}
response = requests.post(url,data=data)
tokens = response.json()

print(tokens)

import json
with open("kakao_code.json", "w") as fp:
    json.dump(tokens,fp)
