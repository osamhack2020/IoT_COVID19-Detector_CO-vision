import cv2
import numpy as np

img = cv2.imread("junha.png")

img = img[500:550, 50:200]

img = cv2.resize(img, None, fx=0.5, fy=0.5)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
adaptive_threshold = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 85, 11)
cv2.imshow('img',adaptive_threshold)
cv2.waitKey(0)
cv2.destroyAllWindows()