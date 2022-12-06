# 가우시안 블러
import cv2
from utils import image_show

image_path = './images/capybara.png'
image = cv2.imread(image_path)

image_g_blur = cv2.GaussianBlur(image, (9,9),0)
image_show(image_g_blur)