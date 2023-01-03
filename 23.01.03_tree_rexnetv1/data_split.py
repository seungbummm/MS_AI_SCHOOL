import matplotlib.pyplot as plt
import glob
import os
from utils import *
from sklearn.model_selection import train_test_split
import cv2

data_dir = './dataset'

# 데이터 분포를 보기 위한 시각화
# data_size_show(data_dir)

def image_split(path):
    file_label = os.listdir(path)
    folder = []
    # for mode in split:
    for label in file_label:
        mode_folder = []
        file_path = glob.glob(os.path.join(path,label,'*.jpg'))
        train_li, val_li = train_test_split(file_path,test_size=0.2, random_state=7777)
        val_li, test_li = train_test_split(val_li,test_size=0.5, random_state=7777)
        
        mode_folder.append(train_li)
        mode_folder.append(val_li)
        mode_folder.append(test_li)
        folder.append(mode_folder)
    return folder
    
'''image_split 구조
    [[[train][val][test]],[[train][val][test]]...]
'''
modes = ['train','val','test']
for label in image_split(data_dir):
    for index, mode in zip(modes,label):
        for path in mode:
            image_name = os.path.basename(path)
            image_name = image_name.replace('.JPG','')
            folder_name = path.split('\\')[1]
            os.makedirs(f'./data/{index}/{folder_name}',exist_ok=True)
            if index=='train':
                folder_path = f'./data/{index}/{folder_name}'
            elif index=='val':
                folder_path = f'./data/{index}/{folder_name}'
            else:
                folder_path = f'./data/{index}/{folder_name}'
            img = cv2.imread(path)
            cv2.imwrite(os.path.join(folder_path, image_name+'.png'),img)