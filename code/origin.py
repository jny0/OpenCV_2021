import cv2
import numpy as np
from matplotlib import pyplot as plt

cap = cv2.VideoCapture("C:\Python\image\myface.mp4") #영상 가져오기 (경로 입력)

while True:  #영상 재생
    retval, frame = cap.read()	# 영상을 한 frame씩 읽어오기 (retval = 정상적으로 읽어왔는지 여부 / frame = 읽어온 프레임)
    if not retval:  #frame 정보를 읽지 못하면 while문 빠져나가기 
        break      
    #hist = cv2.calcHist([frame],[0],None,[256],[0,256]) # 히스토그램 구하기   
    hist, bin = np.histogram(frame.ravel(), 256, [0, 256]) 
    cv2.imshow('frame',frame)	# frame 보여주기

    key = cv2.waitKey(30) # 30ms동안 한 프레임을 보여줌
    if key == 27:   # esc 키(아스키코드 27)를 누르면 while문 종료(재생 창 닫기)
        break

# 그래프 디테일
plt.hist(frame.ravel(), 256, [0,256]); 
plt.title("Original Histogram") #그래프에 제목 넣기
plt.show()


cap.release()	# 영상 사용 종료
cv2.destroyAllWindows() # 모든 창 닫기