import cv2
import numpy as np
from utils import image_show
import matplotlib.pyplot as plt

img_gray = cv2.imread('./images/Billiards.png', cv2.IMREAD_GRAYSCALE)
_, mask = cv2.threshold(img_gray, 230, 255, cv2.THRESH_BINARY_INV)

# 3X3 Kernels
kernel = np.ones((5,5), np.uint8) # uint8 부호 없는 정수
# 결과값
# [[1 1 1]
#  [1 1 1]
#  [1 1 1]]

dilation = cv2.dilate(mask, kernel)
titles = ['image','mask','dilation']
images = [img_gray,mask,dilation]

for i in range(3):
    plt.subplot(1,3,i+1)
    plt.title(titles[i])
    plt.imshow(images[i], 'gray')
    plt.xticks([])
    plt.yticks([])
plt.show()
