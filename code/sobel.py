import cv2
import numpy as np
from matplotlib import pyplot as plt

cap = cv2.VideoCapture("C:\Python\image\myface.mp4") # 영상 가져오기 (경로 입력)

while True:  #영상 재생
    retval, frame = cap.read()
    if not retval: 
        break   

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # 영상 그레이스케일로 변경

    # sobel mask 직접 생성
    gx_k = np.array([[-1,0,1], [-2,0,2],[-1,0,1]])
    gy_k = np.array([[-1,-2,-1],[0,0,0], [1,2,1]])

    edge_gx = cv2.filter2D(gray, -1, gx_k) #필터 적용
    edge_gy = cv2.filter2D(gray, -1, gy_k)

    # OpenCv를 활용
    sobelx = cv2.Sobel(gray, -1, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, -1, 0, 1, ksize=3) 

    cv2.imshow('original', gray) 
    cv2.imshow('Gx edge', edge_gx) 
    cv2.imshow('Gy edge', edge_gy) 
    cv2.imshow('Gx-Gy edge', edge_gx+edge_gy)

    key = cv2.waitKey(30)
    if key == 27:  
        break

cap.release()	# 영상 사용 종료
cv2.destroyAllWindows() # 모든 창 닫기