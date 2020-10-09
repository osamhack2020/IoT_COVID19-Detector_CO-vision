import json
import requests

def getFriendList(accessToken) :
    url = 'https://kapi.kakao.com/v1/api/talk/friends'
    payload = ''
    headers = {
        'Content-Type' : "application/x-www-form-urlencoded",
        'Cache-Control' : "no-cache",
        'Authorization' : "Bearer " + str(accessToken),
    }
    response = requests.request("GET",url,data=payload, headers=headers)
    print(response)
    friend_List = json.loads(((response.text).encode('utf-8')))
    return friend_List

accessToken = 'accessToken'
print(getFriendList(accessToken))