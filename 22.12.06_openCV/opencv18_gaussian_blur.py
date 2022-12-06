# 가우시안 필터 적용
import cv2
import numpy as np
from utils import image_show

image_path = './images/car.jpg'
image = cv2.imread(image_path, 0) # 흑백처리
image_resize = cv2.resize(image, (640,360))

Gaussian_blurred_1 = np.hstack([
    cv2.GaussianBlur(image_resize, (3,3), 0),
    cv2.GaussianBlur(image_resize, (5,5), 0),
    cv2.GaussianBlur(image_resize, (9,9), 0),
])
image_show(Gaussian_blurred_1)
cv2.imwrite('./gaussian_blur.png', Gaussian_blurred_1)