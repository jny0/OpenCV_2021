import cv2
import numpy as np
from matplotlib import pyplot as plt

cap = cv2.VideoCapture("C:\Python\image\myface.mp4") #영상 가져오기 (경로 입력)

while True:  #영상 재생
    retval, frame = cap.read()	# 영상을 한 frame씩 읽어오기 (retval = 정상적으로 읽어왔는지 여부 / frame = 읽어온 프레임)
    if not retval:  #frame 정보를 읽지 못하면 while문 빠져나가기 
        break   

    val = 50.0  # 밝기값

    #opencv활용
    #array = np.full(frame.shape, (val, val, val), dtype=np.uint8) # 모든 값을 val로 초기화
    add_dst = cv2.add(frame, (val, val, val,0))
    sub_dst = cv2.subtract(frame, (val, val, val,0))
    
    #np연산
    bright_dst = np.clip(frame + val, 0, 255).astype(np.uint8) # 원frame에 val 밝기 만큼 더해준 뒤 포화연산 수행
    dark_dst = np.clip(frame - val, 0, 255).astype(np.uint8) # 원frame에 val 밝기 만큼 빼준 뒤 포화연산 수행

    hist = cv2.calcHist([frame],[0],None,[256],[0,256]) 
    hist_bright = cv2.calcHist([bright_dst],[0],None,[256],[0,256]) 
    hist_dark = cv2.calcHist([dark_dst],[0],None,[256],[0,256]) 

    cv2.imshow('frame',frame)	# frame 보여주기
    cv2.imshow('bright',bright_dst)
    cv2.imshow('dark',dark_dst)

    key = cv2.waitKey(30) # 30ms동안 한 프레임을 보여줌
    if key == 27:   # esc 키(아스키코드 27)를 누르면 while문 종료(재생 창 닫기)
        break



    
# 그래프
plt.figure(1)
plt.plot(hist)
plt.title("Original Histogram")

plt.figure(2)
plt.plot(hist_bright)
plt.title("Bright Histogram")

plt.figure(3)
plt.plot(hist_dark)
plt.title("Dark Histogram")



plt.draw()
plt.show()
# plt.savefig("C:\Python\image\origin_histogram.png", dpi=300)

cap.release()	# 영상 사용 종료
cv2.destroyAllWindows() # 모든 창 닫기