import cv2
from utils import image_show

image_path = './images/capybara.png'
image_gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

image_10X10 = cv2.resize(image_gray, (10,10))
image_10X10.flatten() # 이미지 데이터를 1차원 벡터로 변환
image_show(image_10X10)