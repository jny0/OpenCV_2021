import cv2
import numpy as np
from matplotlib import pyplot as plt

src = cv2.imread("C:\Python\image\\face2.jpg", cv2.IMREAD_GRAYSCALE)
src = cv2.imread("C:\Python\image\\face2.jpg")

    
dst = cv2.normalize(src, None, 0, 255, cv2.NORM_MINMAX) # 히스토그램 스트레칭은 NORM_MINMAX

# 넘파이로 히스토그램 스트레칭 구현
gmin = np.min(src)
gmax = np.max(src)
dst2 = np.clip((src - gmin) * 255. / (gmax - gmin), 0, 255).astype(np.uint8)

alpha = 0.3
func = (1+alpha) * src - (alpha * 128)
dst3 = np.clip(func, 0, 255).astype(np.uint8)


hist = cv2.calcHist([src],[0],None,[256],[0,256]) 

hist_str_cv = cv2.calcHist([dst],[0],None,[256],[0,256]) 
hist_str_np = cv2.calcHist([dst2],[0],None,[256],[0,256]) 
hist_str_3 = cv2.calcHist([dst3],[0],None,[256],[0,256]) 

# cv2.imshow('src', src)
# cv2.imshow('dst', dst)
# cv2.imshow('dst2', dst2)
# cv2.imshow('dst3',dst3)

# plt.figure(1)
# plt.plot(hist)
# plt.title("Original Histogram")

# plt.figure(2)
# plt.plot(hist_str_cv)
# plt.title("norm_cv Histogram")

# plt.figure(3)
# plt.plot(hist_str_np)
# plt.title("norm_np Histogram")

# plt.figure(4)
# plt.plot(hist_str_3)
# plt.title("norm_3 Histogram")

font1 = {'size': 17   }


plt.figure(1)
plt.imshow(cv2.cvtColor(dst2, cv2.COLOR_BGR2RGB),'gray'), plt.title("Stretching",fontdict=font1,pad=10)
plt.figure(2)
plt.plot(hist_str_np),plt.title("Stretching Histogram"), plt.xlabel("pixel",loc='right'),plt.ylabel("frequency",loc='top',labelpad=20)


plt.show()


cv2.waitKey()


cv2.destroyAllWindows()

