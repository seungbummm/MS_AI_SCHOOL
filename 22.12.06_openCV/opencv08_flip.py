import cv2

image_path = './images/capybara.png'
image = cv2.imread(image_path)

# 1 좌우반전 0 상하반전
dst_temp1 = cv2.flip(image, 1)
dst_temp2 = cv2.flip(image, 0)

cv2.imshow('dst_temp1', dst_temp1)
cv2.imshow('dst_temp2', dst_temp2)
cv2.waitKey(0)