import cv2
import numpy as np
from utils import image_show

image = cv2.imread('./images/car.jpg')

filter = np.array([[27,-7,-25],[4,-1,3],[-1,-1,2]])
custom_image_filter = cv2.filter2D(image,-1,filter)
image_show(custom_image_filter)