# 라즈베리 파이에서 실행할 코드
# 라즈베리 파이에서 받아온 이미지를 컴퓨터나 노트북에 넘겨줌

import socket
import time
from imutils.video import VideoStream
import imagezmq

sender = imagezmq.ImageSender(connect_to='tcp://192.168.0.1:5555')

rpi_name = socket.gethostname() # 라즈베리 파이가 여러개 있을 수도 있어서 이름을 가져옴

picam = VideoStream(usePiCamera=True).start()
time.sleep(2.0)  # warm up 시간

while True:  # 이미지를 보냄 Ctrl-C 하면 중지
    image = picam.read()
    sender.send_image(rpi_name, image)