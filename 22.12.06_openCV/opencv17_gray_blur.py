# 기본적인 이미지 처리 기술을 이용한 이미지 선명화 - 1
import cv2
import numpy as np
from utils import image_show

image_path = './images/car.jpg'
image = cv2.imread(image_path, 0) # 흑백처리
image_resize = cv2.resize(image, (640,360))
print(image.shape)

blurred_1 = np.hstack([     #hstack : 배열을 가로로 이어붙임
    cv2.blur(image_resize,(3,3)),
    cv2.blur(image_resize, (5,5)),
    cv2.blur(image_resize, (9,9))
])
cv2.imshow('show',blurred_1) # blur별 이미지 출력
cv2.imwrite('./blur.png', blurred_1)
cv2.waitKey(0)