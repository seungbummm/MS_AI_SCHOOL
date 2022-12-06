import cv2

image_path = './images/capybara.png'
image = cv2.imread(image_path)
img90 = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE) 
#CLOCKWISE: 시계방향
img180 = cv2.rotate(image, cv2.ROTATE_180) # 180도
img270 = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE) 
#COUNTERCLOCKWISE : 반시계방향

cv2.imshow('original', image)
cv2.imshow('rotate90', img90)
cv2.imshow('rotate180', img180)
cv2.imshow('rotate270', img270)
cv2.waitKey(0)