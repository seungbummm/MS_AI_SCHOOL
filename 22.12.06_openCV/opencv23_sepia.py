import cv2
import numpy as np
from utils import image_show

image = cv2.imread('./images/car.jpg')

# 세피아 효과 필터 (노을 이미지 만들 때 사용)
filter = np.array([[0.272,0.534,0.131],
[0.349, 0.686, 0.168],[0.393,0.769,0.189]])

sepia_img = cv2.transform(image,filter)
image_show(sepia_img)
cv2.imwrite('./sepia.png',sepia_img)
