from torch.utils.data import DataLoader,random_split
from torch.utils.data import Dataset as BaseDataset
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import glob
from PIL import Image
from torchvision import transforms

import warnings
warnings.filterwarnings("ignore")




class SegData(BaseDataset):
    def __init__(self, img_path, masks_path,transform=None):
        self.masks_path = masks_path
        self.img_path = img_path
        self.imgs = glob.glob(os.path.join(img_path, '*.png'))
        self.transform= transform
        

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        img_path = self.imgs[index]
        #PROBLEM VAR MI KONTROL ET
        #EĞİTİM ICIN PROBLEM VAR MI
        if(img_path.split('/')[-3]=="Test"):
            mask_path = os.path.join(self.masks_path, 'test_mask_' + img_path.split('/')[-1].split('_')[-1])
        else:
            mask_path = os.path.join(self.masks_path, 'train_mask_' + img_path.split('/')[-1].split('_')[-1])
        
        img_arr=cv2.imread(img_path)
        img_arr=cv2.cvtColor(img_arr,cv2.COLOR_BGR2RGB) #ASLINDA BGR DONDURUYOR RGB CEVIRIYORUZ. OPENCV PROBLEM
        mask_arr=cv2.imread(mask_path)
        
        car_color = (142, 0, 0)
        human_colors = [(60, 20, 220), (0, 0, 255)]
        
        zeros_mask=np.zeros(mask_arr.shape[:2])
        
        # Görüntüyü düzenle
        mask_car = np.all(mask_arr == np.array(car_color), axis=2)
        mask_human = np.any(np.all(mask_arr[:, :, None, :] == np.array(human_colors)[None, None, :, :], axis=3), axis=2)

        zeros_mask[mask_car] = 1
        zeros_mask[mask_human] = 2 

        #mask_arr[~(mask_car | mask_human)] = 0
        #mask_arr=cv2.cvtColor(mask_arr,cv2.COLOR_RGB2GRAY)
        
        
   
        if self.transform:
            image = self.transform(img_arr)
            mask =  self.transform(zeros_mask)

        return image, mask



class KittiDatasetLoader():
    def __init__(self, data_dir, label_data_dir,batch_size):
        self.data_dir = data_dir
        self.label_dir= label_data_dir
        self.batch_size=batch_size
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((512, 512)),
        ])
        
    def get_data(self):
        custom_dataset = SegData(self.data_dir,self.label_dir,transform=self.transform)
        
        total_size = len(custom_dataset)
        train_size = int(0.9 * total_size)  # Örnek olarak %80 train, %20 validation
        val_size = total_size - train_size

        train_dataset, val_dataset = random_split(custom_dataset, [train_size, val_size])
        
        train_dataloader =   DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, drop_last=True,num_workers=4)
        val_dataloader   =   DataLoader(val_dataset, batch_size=1, shuffle=False,num_workers=4)
        
        return train_dataloader,val_dataloader,train_size,val_size

