import cv2
from utils import image_show
import numpy as np
image_path = './images/capybara.png'
image = cv2.imread(image_path)

# 10X10 픽셀 크기로 변호나
image_color_10X10 = cv2.resize(image, (10,10))
image_shape_info = image_color_10X10.flatten().shape
image_color_10X10.flatten()
print('image_shape_info', image_shape_info)
#image_shape_info (300,)
image_show(image_color_10X10)

# image 225X255 픽셀 크기로 변환
image_color_225X255 = cv2.resize(image, (225,255))
image_color_225X255.flatten()
image_show(image_color_225X255)

# x = np.array([[51,40],[14,19],[10,7]])
# x = x.flatten()
# print(x)
# 결과 [51 40 14 19 10 7]