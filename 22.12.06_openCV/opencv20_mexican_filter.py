import cv2
import numpy as np
from utils import image_show

image = cv2.imread('./images/car.jpg')

# creating maxican hat filter
filter = np.array([[0,0,-1,0,0],[0,-1,-2,-1,0],
[-1,-2,16,-2,-1],[0,-1,-2,-1,0],[0,0,-1,0,0]])
mexican_hat_img = cv2.filter2D(image,-1,filter)
image_show(mexican_hat_img)
cv2.imwrite('./mexican_5X5.png',mexican_hat_img)
