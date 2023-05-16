import cv2
import numpy as np
from matplotlib import pyplot as plt

cap = cv2.VideoCapture("C:\Python\image\myface.mp4") #영상 가져오기 (경로 입력)

while True:  #영상 재생
    retval, frame = cap.read()
    if not retval: 
        break   

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # 영상 그레이스케일로 변경

    f_min = np.min(gray) # 최소픽셀값
    f_max = np.max(gray) # 최대픽셀값
    dst = np.clip((gray - f_min) * 255. / (f_max - f_min), 0, 255).astype(np.uint8) #스트레칭 연산 수행

    hist = cv2.calcHist([gray],[0],None,[256],[0,256])  # 그레이스케일 히스토그램
    hist_stret = cv2.calcHist([dst],[0],None,[256],[0,256]) # 스트레칭 히스토그램
    
    cv2.imshow('gray',gray)	# 결과 영상
    cv2.imshow('stret',dst)

    key = cv2.waitKey(30)
    if key == 27:  
        break


# 그래프
plt.figure(1)
plt.plot(hist)
plt.title("Original Histogram")

plt.figure(2)
plt.plot(hist_stret)
plt.title("Stretching Histogram")

plt.show()

cap.release()	# 영상 사용 종료
cv2.destroyAllWindows() # 모든 창 닫기