import torch
import torch.nn as nn

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim import Adam

class OptimizerFunctions():
    def __init__(self,model,learning_rate,optimizer_type='ADAM'):
        self.optimizer_type=optimizer_type
        self.model=model
        self.lr=learning_rate

    def get_optimizer(self):
        if self.optimizer_type == 'ADAMW':
            return AdamW(self.model.parameters(),self.lr,weight_decay=1e-5)
        
        if self.optimizer_type == 'ADAM':
            return Adam(self.model.parameters(),self.lr,weight_decay=1e-5)
        
        
        
        else:
            raise ValueError("Geçersiz tip")

