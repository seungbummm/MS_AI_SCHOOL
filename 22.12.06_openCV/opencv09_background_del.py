import cv2
import numpy as np 
from utils import image_show

image_path = './images/capybara.png'

image = cv2.imread(image_path)
image_rgb = cv2.imread(image_path)

# 사각형 좌표 : 사각점의 x y 넓이 높이
rectangle = (0,100,800,1000)

# 초기 마스크 생성
mask = np.zeros(image_rgb.shape[:2], np.uint8)

# grabCut에 사용할 임시 배열 생성
bgdModel = np.zeros((1, 65), np.float64)
fgdModel = np.zeros((1, 65), np.float64)

# grabCut 실행
# image -> 원본 이미지, bgdModel -> 배경을 위한 임시 배열 fgdModel -> 전경배경,
# 5 -> 반복횟수 cv2.GC_INIT_WITH_RECT -> 사각형 초기화
cv2.grabCut(image, mask, rectangle, bgdModel,
            fgdModel, 5, cv2.GC_INIT_WITH_RECT)

#배경인곳은 0, 그 외에는 1로 설정한 마스크 생성

mask_2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
image_rgb_nobg = image_rgb * mask_2[:,:,np.newaxis]
image_show(image_rgb_nobg)