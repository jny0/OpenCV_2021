import cv2
import numpy as np
from matplotlib import pyplot as plt


img = cv2.imread("C:\Python\image\myface.png")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
height, width = gray.shape 

# laplacian mask 생성
mask = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]]) 
laplacian = cv2.filter2D(gray, -1, mask) # 필터 적용

# OpenCv를 활용
laplacian_opencv = cv2.Laplacian(gray, -1) 

gaussian_mask = np.array([[1 / 16, 1 / 8, 1 / 16], [1 / 8, 1 / 4, 1 / 8], [1 / 16, 1 / 8, 1 / 16]])

gaussian_out = cv2.filter2D(gray, -1, gaussian_mask) 
LoG = cv2.filter2D(gaussian_out, -1, mask) 

cv2.imshow('original', gray) 
cv2.imshow('laplacian1', laplacian.astype(np.float))
cv2.imshow('laplacian2', laplacian_opencv.astype(np.float))
cv2.imshow('LoG', LoG.astype(np.float))
  

cv2.waitKey(0)