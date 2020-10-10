import requests

headers = {
    'Authorization': 'Bearer {USER_ACCESS_TOKEN}',
}

data = {
  'template_object': '{ "object_type": "text", "text": "It is text", "link": { "web_url": "https://developers.kakao.com", "mobile_web_url": "https://developers.kakao.com" }, "button_title": "Check" }'
}

response = requests.post('https://kapi.kakao.com/v2/api/talk/memo/default/send', headers=headers, data=data)
