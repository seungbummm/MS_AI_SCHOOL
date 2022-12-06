import cv2
import numpy as np

image_path = './images/capybara.png'
image = cv2.imread(image_path)
channels = cv2.mean(image)
print("Channels >> ", channels)
observation = np.array([(channels[2], channels[1], channels[0])])
print(observation)