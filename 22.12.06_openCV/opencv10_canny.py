import cv2
import numpy as np
import matplotlib.pyplot as plt
from utils import image_show

image_path = './images/test.jpg'

# 경계선 찾기, canny를 할 때에는 GRAYSCALE
image_gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

median_intersity = np.median(image_gray)
print(median_intersity) # 중간값 : 242.0

# 중간 픽셀 강도에서 위아래 1표준편차 떨어진 값을 임계값으로 설정
lower_threshold = int(max(0,(1.0-0.2)*median_intersity))
upper_threshold = int(min(255,(1.0+0.2)*median_intersity))

# Canny edge Detection 적용
image_canny = cv2.Canny(image_gray, lower_threshold, upper_threshold)
image_show(image_canny)