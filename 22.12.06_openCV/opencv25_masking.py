import cv2
import matplotlib.pyplot as plt

# image loading and input image -> gray
image = cv2.imread('./images/Billiards.png',cv2.IMREAD_GRAYSCALE)

# 임계값 연산자의 출력을 마스크라는 변수에 저장,
# 저장하지 않는 출력은 _을 통해 사용X
# 230보다 작으면 모든 값은 흰색 처리, 230 보다 큰 값은 검은색 처리
_, mask = cv2.threshold(image, 230, 255, cv2.THRESH_BINARY_INV)

titles = ['image','mask']
images = [image, mask]

for i in range(2):
    plt.subplot(1, 2, i+1)
    plt.imshow(images[i], 'gray')
    plt.title(titles[i])
    plt.xticks([])
    plt.yticks([])
plt.show()