import torch
import os
import numpy as np
import matplotlib.pyplot as plt
from segmentation.data.dataloader import SegData
import torchvision
from torchvision import transforms
import configparser
from torch.nn.functional import one_hot
import cv2
import warnings
warnings.filterwarnings("ignore")

import requests
requests.packages.urllib3.disable_warnings()
import ssl

#hata fix
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    # Legacy Python that doesn't verify HTTPS certificates by default
    pass
else:
    # Handle target environment that doesn't support HTTPS verification
    ssl._create_default_https_context = _create_unverified_https_context


import time


class SegmentationInference:
    def __init__(self, config_path="./segmentation/config/inference_config.yaml"):
        self.config = configparser.ConfigParser()
        self.config.read(config_path)
        self.num_classes=int(self.config.get('data', 'num_classes'))
        self.test_path = self.config.get('path_directory', 'test_path')
        self.test_mask_path = self.config.get('path_directory', 'test_mask_path')
        self.model_path = self.config.get('model', 'model_path')
        self.device = self.config.get('model', 'device')
        
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((512, 512)),
        ])
        
        self.kernel=np.ones((3, 3), np.uint8) 

        self.iteration=3

        self.bbox_dict = {}

    def visualize(self, **images):
        n = len(images)
        plt.figure(figsize=(16, 16))
        for i, (name, image) in enumerate(images.items()):
            plt.subplot(1, n, i + 1)
            plt.xticks([])
            plt.yticks([])
            plt.title(' '.join(name.split('_')).title())
            plt.imshow(image)
        plt.show()
    

    def __monitoring(self,image,predict):
        
        #predict
        
        font=cv2.FONT_HERSHEY_SIMPLEX
        font_scale=0.7
        thickness=2

        for i in predict['car']:
            x,y,w,h=i
            pt1 = (x, y)
            pt2 = (x+ w, y+ h)
            #start_point_putText = x, y-5
            cv2.rectangle(image,pt1,pt2, (128, 0, 0), 2) #BBOx
            text_size = cv2.getTextSize("car", font, font_scale, thickness)[0]
            cv2.rectangle(image,(x,y),(x+text_size[0],(y-text_size[1])+2), (128 ,0, 0), -1) #CLASS TEXT
            text_position = (x, y)
            cv2.putText(image, 'car', text_position, cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)
        
        for i in predict['person']:
            x,y,w,h=i
            pt1 = (x, y)
            pt2 = (x+ w, y+ h)
            #start_point_putText = x , (y - 5)
            cv2.rectangle(image,pt1,pt2, (0, 0, 255), 2)
            text_size = cv2.getTextSize("person", font, font_scale, thickness)[0]
            cv2.rectangle(image,(x,y),(x+text_size[0],(y-text_size[1])+2), (0 ,0, 255), -1) #CLASS TEXT
            text_position = (x,y)
            cv2.putText(image, 'person', text_position, cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)

        return image
        

    def __connected_component_and_detection(self,mask_image:np.ndarray,image:np.ndarray):
        
        mask_dilation = cv2.dilate(mask_image,self.kernel,iterations = self.iteration)
     
        original_image=image.copy()
        
        tensor_mask_dilation=(torch.tensor(mask_dilation)).long()

        one_hot_mask=one_hot(tensor_mask_dilation,self.num_classes) #return 512,512,1,num_classes -- > squeeze(2) 512,512,num_classes int64


        list_of_bbox=[]

    
        for counter_classes in range(1,self.num_classes): #i=1, i=2
            
            input_for_component_mask=np.array(one_hot_mask[:,:,counter_classes].unsqueeze(2),dtype=np.uint8)
            
            output = cv2.connectedComponentsWithStats(
            input_for_component_mask, 4, cv2.CV_32S)

            (numLabels, labels, stats, centroids) = output

     
            list_of_bbox.clear()
          
            for i in range(1, numLabels):
                x = stats[i, cv2.CC_STAT_LEFT]
                y = stats[i, cv2.CC_STAT_TOP]
                w = stats[i, cv2.CC_STAT_WIDTH]
                h = stats[i, cv2.CC_STAT_HEIGHT]
                bbox=(x,y,w,h) #bbox analys
               

                #kucuk maskeleri alma
                keepWidth = w > 5 and w < 1000
                keepHeight = h > 5 and h < 1000
             
                
                if all((keepWidth, keepHeight)):
                    list_of_bbox.append(bbox)


            if(counter_classes==1): #back groundu hesaba katmıyoruz
                self.bbox_dict["car"] = list_of_bbox.copy()
                        
            elif(counter_classes==2):
                self.bbox_dict["person"] = list_of_bbox.copy()
    

        return self.__monitoring(original_image,self.bbox_dict),mask_image
    
    def run(self):
        model = torch.load(self.model_path).to(self.device)
        

        for i in range((len(os.listdir(self.test_path)) - 1)):
            
            image, gt_mask = SegData(self.test_path, self.test_mask_path, self.transform).__getitem__(i) #fixlenebilir demiştik dogrudan galeriden cek.
            #İMAGE ALMA YONTEMI DEGISEBILIR
            
            gt_mask = gt_mask.permute(1, 2, 0)

            x_tensor = image.to(self.device).unsqueeze(0).float()
            mask = model.predict(x_tensor).squeeze(0)

            if self.device == "cpu":
                output_predictions = torch.argmax(torch.softmax(mask, dim=0), dim=0).unsqueeze(2)
            else:
                output_predictions = torch.argmax(torch.softmax(mask, dim=0), dim=0).cpu().unsqueeze(2)

            array_output_predictions=np.array(output_predictions,np.uint8) #tensor to np.array dtype uint8
            image=np.array(image.permute(1, 2, 0),np.float32) #tensor to np.array dtype in64
            
            image=(image * 255).astype(np.uint8)
           
            output,mask_image=self.__connected_component_and_detection(array_output_predictions,image)
        
            
            self.visualize(
                Mask_Predict=output_predictions,
                Classification_Predict=output
            )



if __name__ == "__main__":
    inference_instance = SegmentationInference()
   
    inference_instance.run()


