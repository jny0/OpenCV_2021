import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread("C:\Python\image\myface.png")
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  

# 소벨 커널을 직접 생성해서 엣지 검출 ---①
## 소벨 커널 생성
gx_k = np.array([[-1,0,1], [-2,0,2],[-1,0,1]])
gy_k = np.array([[-1,-2,-1],[0,0,0], [1,2,1]])
## 소벨 필터 적용
edge_gx = cv2.filter2D(img, -1, gx_k)
edge_gy = cv2.filter2D(img, -1, gy_k)

# 소벨 API를 생성해서 엣지 검출
sobelx = cv2.Sobel(img, -1, 1, 0, ksize=3)
sobely = cv2.Sobel(img, -1, 0, 1, ksize=3) 

# 결과 출력
# merged1 = np.hstack((img, edge_gx, edge_gy, edge_gx+edge_gy))
# merged2 = np.hstack((img, sobelx, sobely, sobelx+sobely))
# merged = np.vstack((merged1, merged2))
# cv2.imshow('sobel', merged)

font1 = {'size': 35 }

plt.figure(1)
plt.subplot(121),plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB),'gray'), plt.title("Original",fontdict=font1,pad=10)
plt.subplot(122),plt.imshow(cv2.cvtColor(edge_gx, cv2.COLOR_BGR2RGB),'gray'), plt.title("Gx Edge",fontdict=font1,pad=10)
plt.figure(2)
plt.subplot(121),plt.imshow(cv2.cvtColor(edge_gy, cv2.COLOR_BGR2RGB),'gray'), plt.title("Gy Edge",fontdict=font1,pad=10)
plt.subplot(122),plt.imshow(cv2.cvtColor(edge_gx+edge_gy, cv2.COLOR_BGR2RGB),'gray'), plt.title("Gx-Gy Edge",fontdict=font1,pad=10)
plt.show()

cv2.waitKey(0)
cv2.destroyAllWindows()