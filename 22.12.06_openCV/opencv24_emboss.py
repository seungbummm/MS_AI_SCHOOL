import cv2
import numpy as np
from utils import image_show

image = cv2.imread('./images/car.jpg')
# 엠보싱 효과

filter1 = np.array([[0,1,0],[0,0,0],[0,-1,0]])
filter2 = np.array([[-1,-1,0],[-1,0,1],[0,1,1]])
emboss = cv2.filter2D(image,-1,filter1) # 이상태로 출력하면 잘 안보인다.
emboss += 128
emboss2 = cv2.filter2D(image,-1,filter2)
emboss2 += 128
img = np.hstack([emboss, emboss2])
image_show(img)