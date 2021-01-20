# 라즈베리 파이에서 실행할 코드
# 라즈베리 파이에서 받아온 이미지를 컴퓨터나 노트북에 넘겨줌

import socket
import time
from imutils.video import VideoStream
import imagezmq
from flirpy.camera.lepton import Lepton

thermal_camera = Lepton()

sender = imagezmq.ImageSender(connect_to='tcp://192.168.0.1:5555') # 부여된 개별 주소에 맞게 변경

#rpi_name = socket.gethostname() # 라즈베리 파이가 여러개 있을 수도 있어서 이름을 가져옴

picam = VideoStream(usePiCamera=True).start()
time.sleep(2.0)  # warm up 시간

while True:  # 이미지를 보냄 Ctrl-C 하면 중지
    img = picam.read()
    sender.send_image('visible_image', img) #가시광선 카메라 이미지 전송

    thermal_image_data = thermal_camera.grab()
    sender.send_image('thermal_image', thermal_image_data) #적외선 카메라 이미지 전송

thermal_camera.close()