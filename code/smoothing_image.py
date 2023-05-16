import cv2 
import numpy as np 
from matplotlib import pyplot as plt


img = cv2.imread("C:\Python\image\myface.png") 


gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
averaging_mask = np.ones((5, 5), dtype=np.float64) / 25.
gaussian_mask = np.array([[1 / 16, 1 / 8, 1 / 16], [1 / 8, 1 / 4, 1 / 8], [1 / 16, 1 / 8, 1 / 16]])

averaging_out = cv2.filter2D(gray, -1, averaging_mask) 
gaussian_out = cv2.filter2D(gray, -1, gaussian_mask) 

blurring_opencv = cv2.blur(gray, (5, 5), anchor=(-1, -1), borderType=cv2.BORDER_DEFAULT)

font1 = {'size': 17   }

cv2.imshow('original', gray) 
cv2.imshow('averaging', averaging_out) 
cv2.imshow('gaussian', gaussian_out) 
cv2.imshow('blurring_opencv', blurring_opencv)

cv2.waitKey(0)