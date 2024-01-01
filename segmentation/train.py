import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms

import numpy as np
import cv2
from tqdm import tqdm
import configparser
import matplotlib.pyplot as plt
import os

from custom_utils.loss.custom_loss import FocalLoss
from custom_utils.optimizer.optimizer import OptimizerFunctions
from data.dataloader import KittiDatasetLoader
from segmentation_models_pytorch import utils

import warnings
warnings.filterwarnings("ignore")

import ssl
import requests

requests.packages.urllib3.disable_warnings()

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    # Legacy Python that doesn't verify HTTPS certificates by default
    pass
else:
    # Handle target environment that doesn't support HTTPS verification
    ssl._create_default_https_context = _create_unverified_https_context



class SegmentationTrain:
    def __init__(self, config_path="./config/train_config.yaml"):
        
        self.config = configparser.ConfigParser()
        self.config.read(config_path)

        self.train_path = self.config.get('path_directory', 'train_path')
        self.train_mask_path = self.config.get('path_directory', 'train_mask_path')
        self.model_save_path = self.config.get('path_directory', 'model_save_path')

        self.device = self.config.get('model', 'device')
        self.encoder=self.config.get('model', 'encoder')
        self.encoder_weights=self.config.get('model', 'encoder_weights')
        
        self.num_classes=int(self.config.get('train_parameters', 'num_classes'))
        self.epochs=int(self.config.get('train_parameters', 'epochs'))
        self.batch_size=int(self.config.get('train_parameters', 'batch_size'))
        self.optimizer_name=self.config.get('train_parameters', 'optimizer')
        self.learning_rate=float(self.config.get('train_parameters', 'learnig_r'))
        self.loss_func=self.config.get('train_parameters', 'loss')
        self.criterion=None

        self.__model_init()
        self.__data_load_init()
        

    
    def __model_init(self):

        self.model = smp.Unet(
            encoder_name=self.encoder,       
            encoder_weights=self.encoder_weights,   
            in_channels=3,                  
            classes=self.num_classes #model channel output = class number
        )
        
        self.model.to(self.device)
        
        if(self.loss_func=="FocalLoss"):
            self.criterion=FocalLoss() 
        else:
            self.criterion=nn.CrossEntropyLoss()
        
        self.optimizer=OptimizerFunctions(self.model,self.learning_rate,self.optimizer_name).get_optimizer()
        
        print("**********")
        print("Device -> ",self.device)
        print("Loss ->",self.criterion)
        print("Epochs ->",self.epochs)
        print("Batch Size ->",self.batch_size)
        print("Learning Rate ->",self.learning_rate)
        print("Optimizer ->",self.optimizer)
        print("**********")

    def __data_load_init(self):
        self.train_dataset , self.val_dataset, self.len_data, self.len_val= KittiDatasetLoader(self.train_path,self.train_mask_path,self.batch_size).get_data()


    def __train_fit(self,model, dataloader, data, optimizer, criterion):
        print('-------------Training------------------')
        
        self.model.train()
        train_running_loss = 0.0
        
        counter = 0
        num_batches = int(data / dataloader.batch_size)
        
        for i, d in tqdm(enumerate(dataloader), total=num_batches):
            counter += 1
            image, mask = d[0].to(self.device), d[1].to(self.device)

            optimizer.zero_grad()
            outputs = model(image)
            mask=mask.squeeze(1).long()
            
            loss = criterion(outputs, mask)
            
            train_running_loss += loss.item()
            
            loss.backward()
            optimizer.step()

        train_loss = train_running_loss / counter
        

        return train_loss

    def __validate(self, model, dataloader, data, criterion):
        print('--------------Validating-------------------')
        self.model.eval()
        valid_running_loss = 0.0
        counter = 0
        
        num_batches = int((data) / dataloader.batch_size)
       
        with torch.no_grad():
            for i, data in tqdm(enumerate(dataloader), total=num_batches):
                counter += 1
                image, mask = data[0].to(self.device), data[1].to(self.device)
                outputs = model(image)
                mask=mask.squeeze(1).long()
                loss = criterion(outputs, mask)
                #
                valid_running_loss += loss.item()
        valid_loss = valid_running_loss / counter
        
        return valid_loss
    
    def run(self):
        max_score=torch.inf

        for i in range(0,self.epochs):
            
            train_loss = self.__train_fit(self.model, self.train_dataset, self.len_data, self.optimizer, self.criterion)
            val_loss   = self.__validate(self.model, self.val_dataset, self.len_val, self.criterion)
            
            if(max_score>val_loss):
                max_score=val_loss
                torch.save(self.model, self.model_save_path+'/best_model.pth')
                print("Best Model Kaydedildi")
            else:
                torch.save(self.model, self.model_save_path+'/any_model.pth')
                
            print(f'Epoch {i + 1}/{self.epochs} -> '
                f'Eğitim Kaybı: {train_loss:.4f}' 
                f' --> Doğrulama Kaybı: {val_loss:.4f}')
        torch.save(self.model, self.model_save_path+'/last_model.pth')

if __name__ == "__main__":
    train=SegmentationTrain()
    train.run()


