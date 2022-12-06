import cv2
import numpy as np
import matplotlib.pyplot as plt
from utils import image_show

image_path = './images/capybara.png'

image_gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
max_output_value = 255
neightborhood_size = 101
subtract_from_meam = 5
image_binarized = cv2.adaptiveThreshold(image_gray, 
max_output_value,
cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
cv2.THRESH_BINARY,  #_INV는 반전
neightborhood_size,
subtract_from_meam)
image_show(image_binarized)