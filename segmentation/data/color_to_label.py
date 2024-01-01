import numpy as np
import torch.nn as nn

class Color_to_label_map(nn.Module):
    def __init__(self):
        super(Color_to_label_map, self).__init__()
        self.car_color=(142, 0, 0)
        self.human_colors = [(60, 20, 220), (0, 0, 255)]
        
    
    def forward(self,x):#image 

            #siyas maske oluştur
            self.zeros_mask=np.zeros(x.shape[:2])
            #pixel değerlerine göre 1 ve 2 diye etiket ata
            mask_car = np.all(x == np.array(self.car_color), axis=2)
            mask_human = np.any(np.all(x[:, :, None, :] == np.array(self.human_colors)[None, None, :, :], axis=3), axis=2)
            #maskelemeyi yaparak label_map olustur
            self.zeros_mask[mask_car] = 1
            self.zeros_mask[mask_human] = 2

            return self.zeros_mask



