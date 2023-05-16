import cv2
import numpy as np
from matplotlib import pyplot as plt

cap = cv2.VideoCapture("C:\Python\image\myface.mp4") # 영상 가져오기 (경로 입력)

while True:  #영상 재생
    retval, frame = cap.read()
    if not retval: 
        break   

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # 영상 그레이스케일로 변경

    # laplacian mask 생성
    mask = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]]) 
    laplacian = cv2.filter2D(gray, -1, mask) # 필터 적용

    # OpenCv를 활용
    laplacian_opencv = cv2.Laplacian(gray, -1) 

    gaussian_mask = np.array([[1 / 16, 1 / 8, 1 / 16], [1 / 8, 1 / 4, 1 / 8], [1 / 16, 1 / 8, 1 / 16]])

    gaussian_out = cv2.filter2D(gray, -1, gaussian_mask) 
    LoG = cv2.filter2D(gaussian_out, -1, mask) 

    cv2.imshow('original', gray) 
    cv2.imshow('laplacian1', laplacian.astype(np.float))
    cv2.imshow('laplacian2', laplacian_opencv.astype(np.float))
    cv2.imshow('LoG', LoG.astype(np.float))

    key = cv2.waitKey(30)
    if key == 27:  
        break

cap.release()	# 영상 사용 종료
cv2.destroyAllWindows() # 모든 창 닫기