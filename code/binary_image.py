import cv2
import numpy as np
import matplotlib.pylab as plt

img = cv2.imread('C:\Python\image\\myface.png', cv2.IMREAD_GRAYSCALE) # 이미지를 그레이 스케일로 읽기

threshold = 60 # threshold 값

# NumPy로 이진화
threshold_np = np.zeros_like(img)   # 원 영상과 동일한 크기의 0으로 채워진 frame
threshold_np[ img > threshold] = 255       #  보다 큰 값만 255로 변경

# OpenCV로 이진화
ret, threshold_cv = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY) 


# 출력
# imgs = {'Original': img, 'NumPy':threshold_np, 'OpenCV': threshold_cv}
# for i , (key, value) in enumerate(imgs.items()):
#     plt.subplot(1, 3, i+1)
#     plt.title(key)
#     plt.imshow(value, cmap='gray')
#     #plt.imshow(cv2.cvtColor(value, cv2.COLOR_BGR2RGB),'gray')
#     plt.xticks([]); plt.yticks([])

plt.imshow(threshold_np, cmap='gray')
plt.show()

