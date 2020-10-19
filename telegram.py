import telepot

# bot = co_vision_bot
token = '1130712531:AAE3W0J9Y3s2opGvE_c8My8e96-vhqlLAGE'
mc = '1314303321'
bot = telepot.Bot(token)

f = open('imgs/jun_name1.png','rb')
response = bot.sendPhoto(mc, f)