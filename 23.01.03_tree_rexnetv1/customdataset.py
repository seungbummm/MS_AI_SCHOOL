import glob
import os
from torch.utils.data import Dataset
import cv2

class customdataset(Dataset):
    def __init__(self, file_path, transform = None):
        self.file_path = glob.glob(os.path.join(file_path,'*','*.png'))
        self.transform = transform
        # label dict
        self.folder = os.listdir(file_path)
        self.label_dic = {}
        for index, label in enumerate(self.folder):
            self.label_dic[label] = index
        
        # init에서 처리   
        # self.label_list = []
        # for i in self.file_path:
        #     folder_name = i.split('\\')[1]
        #     label = self.label_dic[folder_name]
        #     self.label_list.append(label)
        
        # self.img_list = []
        # for i in self.file_path:
        #     image = cv2.imread(i)
        #     image = cv2.cvtColor(i,cv2.COLOR_BGR2RGB)
        #     self.img_list.append(image)
        
    def __getitem__(self, index):
        path = self.file_path[index]
        label_temp = path.split('\\')
        label = self.label_dic[label_temp[1]]
        image = cv2.imread(path)
        image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        
        if self.transform is not None:
                image = self.transform(image=image)["image"]
        return image, label
    
    def __len__(self):
        return len(self.file_path)
    
# test = customdataset('./data/train')
# for i in test:
#     print(i)
    