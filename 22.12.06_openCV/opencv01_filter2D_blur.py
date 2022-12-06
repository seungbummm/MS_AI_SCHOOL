# 이미지 blur 처리
# filter 2d() 메소드 사용
import numpy as np
import cv2
from utils import image_show

# 이미지 경로
image_path = './images/capybara.png'
image = cv2.imread(image_path)
print(image)

kernel = np.ones((10,10))/25.0 # 모두 더하면 1이 되도록 정규화
image_kernel = cv2.filter2D(image, -1, kernel)
image_show(image_kernel)