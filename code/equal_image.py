import cv2 
import matplotlib.pyplot as plt 
import numpy as np 


img = cv2.imread('C:\Python\image\\myface.png') 

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
hist = cv2.calcHist([gray],[0],None,[256],[0,256]) # 히스토그램 구하기   

cumsum = hist.cumsum() 
LUT = np.uint8((cumsum - cumsum.min()) * 255 / (cumsum.max() - cumsum.min())) 
equ = LUT[gray] 


equ2 = cv2.equalizeHist(gray) 
hist = cv2.calcHist([equ2],[0],None,[256],[0,256]) 
hist2 = cv2.calcHist([equ],[0],None,[256],[0,256]) 
hist3 = cv2.calcHist([gray],[0],None,[256],[0,256]) 

cv2.imshow("original", gray) 
cv2.imshow('result1', equ) 
cv2.imshow('result2', equ2) 


font1 = {'size': 17   }


plt.figure(1)
plt.imshow(cv2.cvtColor(equ, cv2.COLOR_BGR2RGB)), plt.title("Equalization",fontdict=font1,pad=10)
plt.figure(4)
plt.plot(hist2),plt.title("Equalization Histogram"), plt.xlabel("pixel",loc='right'),plt.ylabel("frequency",loc='top',labelpad=20)



plt.show() 
cv2.waitKey()

