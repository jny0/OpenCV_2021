import cv2
import numpy as np
from matplotlib import pyplot as plt

cap = cv2.VideoCapture("C:\Python\image\myface.mp4") #영상 가져오기 (경로 입력)

while True:  #영상 재생
    retval, frame = cap.read()
    if not retval: 
        break   

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #그레이 스케일 변환
    
    threshold = 150 # threshold 값

    # NumPy로 이진화
    threshold_np = np.zeros_like(frame)   # 원 영상과 동일한 크기의 0으로 채워진 frame
    threshold_np[frame > threshold] = 255  # threshold 보다 큰 값만 255로 변경

    # OpenCV로 이진화
    ret, threshold_cv = cv2.threshold(frame, threshold, 255, cv2.THRESH_BINARY) 

    cv2.imshow('frame',frame)	# 원본영상
    cv2.imshow('binary',threshold_np) #결과영상

    key = cv2.waitKey(30)
    if key == 27: 
        break

