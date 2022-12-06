# 컬러 히스토그램 특성 인코딩
import cv2
import matplotlib.pyplot as plt
import numpy as np

image_path = './images/test.jpg'

image_bgr = cv2.imread(image_path)
image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
features = [] # 특성 값을 담을 리스트
colors = ('r','g','b') # 각 컬러 채널에 대해 히스토그램을 계산
for i, channels in enumerate(colors):
    # cv2.calcHist([이미지], [채널 인덱스], [마스크, 여기선 없어서 None], 
    # #[히스토그램 크기], [범위 : 0부터 256까지])
    histogram = cv2.calcHist([image_rgb],[i],None,[256],[0,256])
    plt.plot(histogram, color = channels)
    plt.xlim([0,256])

plt.show()