import cv2
import numpy as np
from matplotlib import pyplot as plt

cap = cv2.VideoCapture("C:\Python\image\myface.mp4") #영상 가져오기 (경로 입력)

while True:  #영상 재생
    retval, frame = cap.read()	# 영상을 한 frame씩 읽어오기 (retval = 정상적으로 읽어왔는지 여부 / frame = 읽어온 프레임)
    if not retval:  #frame 정보를 읽지 못하면 while문 빠져나가기 
        break   

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #그레이 스케일 변환

    # openCV 활용
    dst = cv2.equalizeHist(frame)

    # 수식활용
    hist, bins = np.histogram(frame.flatten(), 256,[0,255]) #그레이 스케일의 히스토그램
    cumsum = hist.cumsum() #각 bin의 누적합 계산
    
    cumsum_d = np.ma.masked_equal(cumsum,0) # 속도개선을 위해 0인 부분 제외

    LUT = np.uint8((cumsum_d - cumsum_d.min()) * 255 / (cumsum_d.max() - cumsum_d.min())) #equalization
    
    LUT = np.ma.filled(LUT,0).astype('uint8')     # 마스크 처리를 했던 부분을 다시 0으로 변환
    
    equ = LUT[frame] #결과 


    hist = cv2.calcHist([frame],[0],None,[256],[0,256]) 
    #hist_dst = cv2.calcHist([dst],[0],None,[256],[0,256]) 
    hist_equ = cv2.calcHist([equ],[0],None,[256],[0,256]) 
    
    cv2.imshow('frame',frame)	# frame 보여주기
    #cv2.imshow('equ_opencv',dst)
    cv2.imshow('Equalization',equ)

    key = cv2.waitKey(30)
    if key == 27: 
        break


# 그래프
plt.figure(1)
plt.plot(hist)
plt.title("Original Histogram")

# plt.figure(2)
# plt.plot(hist_dst)
# plt.title("equ_cv Histogram")

plt.figure(3)
plt.plot(hist_equ)
plt.title("Equalization Histogram")

plt.draw()
plt.show()
# plt.savefig("C:\Python\image\origin_histogram.png", dpi=300)

cap.release()	# 영상 사용 종료
cv2.destroyAllWindows() # 모든 창 닫기