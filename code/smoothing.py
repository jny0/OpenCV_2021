import cv2
import numpy as np
from matplotlib import pyplot as plt

cap = cv2.VideoCapture("C:\Python\image\myface.mp4") # 영상 가져오기 (경로 입력)

while True:  #영상 재생
    retval, frame = cap.read()
    if not retval: 
        break   

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # 영상 그레이스케일로 변경

    #mask 설정
    blurring_mask = np.ones((5, 5), dtype=np.float64) / 25. 
    smoothing_mask = np.array([[1 / 16, 1 / 8, 1 / 16], [1 / 8, 1 / 4, 1 / 8], [1 / 16, 1 / 8, 1 / 16]])

    blurring_out = cv2.filter2D(gray, -1, blurring_mask) # 필터 적용
    smoothing_out = cv2.filter2D(gray, -1, smoothing_mask) 

    #OpenCV 활용
    blurring_opencv = cv2.blur(gray, (5, 5), anchor=(-1, -1), borderType=cv2.BORDER_DEFAULT)

    cv2.imshow('original', gray) 
    cv2.imshow('blurring', blurring_out) 
    cv2.imshow('smoothing', smoothing_out) 
    cv2.imshow('blurring_opencv', blurring_opencv)

    key = cv2.waitKey(30)
    if key == 27:  
        break

cap.release()	# 영상 사용 종료
cv2.destroyAllWindows() # 모든 창 닫기