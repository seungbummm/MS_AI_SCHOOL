import cv2
import numpy as np
from utils import image_show

image = cv2.imread('./images/car.jpg')
filter = np.array([[-1,-1,-1],[-1,8,-1],[-1,-1,-1]])
mexican_hat_img = cv2.filter2D(image,-1,filter)
image_show(mexican_hat_img)
cv2.imwrite('./mexican_3X3.png',mexican_hat_img)