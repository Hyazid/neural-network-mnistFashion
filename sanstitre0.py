# -*- coding: utf-8 -*-
"""
Created on Fri Jun 19 20:58:20 2020

@author: Mon pc
"""
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch import optim

##import gym
transform =  transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5), (0.5)),])
batch_size =64
trainset = datasets.FashionMNIST('~/.pytorch/F_MNIST_data/',download =True,train=True,transform=transform)
trainloader =torch.utils.data.DataLoader(trainset,batch_size=batch_size,shuffle = True)
testset =datasets.FashionMNIST('~/.pytorch/F_MNIST_data/',download =True,train=True,transform=transform)
testloader=torch.utils.data.DataLoader(testset,batch_size=batch_size,shuffle = True)
class Fashionnetwork(nn.Module):
    
     def  __init__(self):
          super().__init__()
          self.hidden1 =nn.Linear(728, 256)
          self.hidden2 =nn.Linear(256, 128)
          self.output = nn.Linear(128, 10)
          #self.softmax = nn.Softmax(dim=1)
          self.logSoftMax = nn.LogSoftmax()
          self.activation = nn.ReLU()
          
          
     def forward(self, x):
         x=self.hidden1(x)
         x=self.activation(x)
         #seconde layer
         x=self.hidden2(x)
         x=self.activation(x)
         # last layer
         x=self.output(x)
         output  = self.LogSoftmax(x)
         #criterion  =nn.NLLLoss()
         ## pushing the out put to softmax function
         
         
         return output
     
        
model = Fashionnetwork()
criterion = nn.NLLLoss()#loss function 
optimizer =optim.Adam(model.parameters(),lr=3e-1)

print(model)
                   
print(optimizer.defaults)  
epoch = 10
for n in range(epoch):
    running_loss= 0
    for image,label in trainloader:
        optimizer.zero_grad()
        image= image.view(image.shape[0],-1)
        pred=model(image)
        loss =criterion(pred, label)
        loss.backward()
        optimizer.step()
        running_loss +=loss.item()
    else:
        print(f"train loss :{running_loss/len(trainloader):.4f}")

        
        

