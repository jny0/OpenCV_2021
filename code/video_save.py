import cv2
import os
from matplotlib import pyplot as plt
 
path = "C:\Python\image\\"
filePath = os.path.join(path, "myface.mp4")
print(filePath)

if os.path.isfile(filePath):	# 해당 파일이 있는지 확인
    # 영상 객체(파일) 가져오기
    cap = cv2.VideoCapture(filePath)
else:
    print("파일이 존재하지 않습니다.")  
 
# frame 사이즈
frame_size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
              int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
print('frame_size =', frame_size)
 
# 코덱 설정하기
#fourcc = cv2.VideoWriter_fourcc(*'DIVX')  # ('D', 'I', 'V', 'X')
fourcc = cv2.VideoWriter_fourcc(*'XVID')
 
# 이미지 저장하기 위한 영상 파일 생성
out1 = cv2.VideoWriter('C:\Python\image\\result\\record0.mp4',fourcc, 20.0, frame_size)
out2 = cv2.VideoWriter('C:\Python\image\\result\\record1.mp4',fourcc, 20.0, frame_size,isColor=False)
 
while True:
    retval, frame = cap.read()	# 영상을 한 frame씩 읽어오기
    if not retval:
        break   
        
    out1.write(frame)	# 영상 파일에 저장   
    
    # 이미지 컬러 변환
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    hist = cv2.calcHist([frame],[0],None,[256],[0,256]) #그레이 스케일
    colors = ['b', 'g', 'r']
    bgr_planes = cv2.split(frame)
    
    out2.write(gray)	# 영상 파일에 저장        
    equ = cv2.equalizeHist(gray)

    cv2.imshow('frame',frame)	# 이미지 보여주기
    cv2.imshow('gray',gray)      
    cv2.imshow('equ',equ)


    key = cv2.waitKey(10)
    if key == 27:
        break



for (p, c) in zip(bgr_planes, colors):
    hist2 = cv2.calcHist([p], [0], None, [256], [0, 256])
    plt.plot(hist2, color=c)


# plt.plot(hist)
# plt.show()

cap.release()	# 객체 해제
out1.release()
out2.release()
cv2.destroyAllWindows()