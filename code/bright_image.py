import cv2
import numpy as np
from matplotlib.gridspec import GridSpec
from matplotlib import pyplot as plt

frame = cv2.imread("C:\Python\image\myface.png") #영상 가져오기 (경로 입력)

#frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)   
val = 50.0

#opencv 활용
array = np.full(frame.shape, (val, val, val), dtype=np.uint8)
add_dst = cv2.add(frame, (val, val, val, 0))
sub_dst = cv2.subtract(frame, (val, val, val, 0))

#np 연산
add_dst2 = np.clip(frame + val, 0, 255).astype(np.uint8)
sub_dst2 = np.clip(frame - val, 0, 255).astype(np.uint8)

    
hist = cv2.calcHist([frame],[0],None,[256],[0,256]) 
hist_add = cv2.calcHist([add_dst2],[0],None,[256],[0,256]) 
hist_sub = cv2.calcHist([sub_dst2],[0],None,[256],[0,256]) 
    
#cv2.imshow('frame',frame)	# frame 보여주기
# cv2.imshow('add',add_dst)
# cv2.imshow('sub',sub_dst)


font1 = {'size': 17   }

# plt.figure(1)
# plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)), plt.title("Original",fontdict=font1,pad=10)
# plt.figure(4)
# plt.plot(hist),plt.title("Original Histogram"), plt.xlabel("pixel",loc='right'),plt.ylabel("frequency",loc='top',labelpad=20)


plt.figure(2)
plt.subplot(221),plt.imshow(cv2.cvtColor(add_dst, cv2.COLOR_BGR2RGB),'gray'), plt.title("Bright",fontdict=font1,pad=10)
plt.subplot(223),plt.imshow(cv2.cvtColor(sub_dst, cv2.COLOR_BGR2RGB),'gray'), plt.title("Dark",fontdict=font1,pad=10)
plt.subplot(222),plt.plot(hist_add),plt.title("Bright Histogram"), plt.xlabel("pixel",loc='right'),plt.ylabel("frequency",loc='top',labelpad=20)
plt.subplot(224),plt.plot(hist_sub),plt.title("Dark Histogram"), plt.xlabel("pixel",loc='right'),plt.ylabel("frequency",loc='top',labelpad=20)

plt.figure(1)
plt.plot(hist_add),plt.title("Bright Histogram"), plt.xlabel("pixel",loc='right'),plt.ylabel("frequency",loc='top',labelpad=20)
plt.figure(2)
plt.plot(hist_sub),plt.title("Dark Histogram"), plt.xlabel("pixel",loc='right'),plt.ylabel("frequency",loc='top',labelpad=20)

plt.figure(3)
plt.subplot(121),plt.plot(hist_add),plt.title("Bright Histogram"), plt.xlabel("pixel",loc='right'),plt.ylabel("frequency",loc='top',labelpad=20)
plt.subplot(122),plt.plot(hist_sub),plt.title("Dark Histogram"), plt.xlabel("pixel",loc='right'),plt.ylabel("frequency",loc='top',labelpad=20)



plt.draw()
plt.show()

cv2.waitKey()
cv2.destroyAllWindows()