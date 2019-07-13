#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 13 14:39:25 2019

@author: gaoyi
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 13 12:46:36 2019

@author: gaoyi
"""

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
device = torch.device("cuda" if torch.cuda.is_available()else "cpu")

#%%
data_dir = '/homes/gaoyi/data/powerarena'
model_use ='vgg19'
def load_split_train_test(datadir, size=64, valid_size = .2, test_size = .2):
    t =  transforms.Compose([transforms.RandomResizedCrop(224),
                        transforms.RandomRotation(30),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        transforms.Normalize([0.485, 0.456, 0.406],                      
                                             [0.229, 0.224, 0.225])])
    
    train_data = datasets.ImageFolder(data_dir,transform=t)
    val_data = datasets.ImageFolder(data_dir,transform=t)
    test_data = datasets.ImageFolder(data_dir,transform=t)
    
    num_train = len(train_data)
    indices = list(range(num_train))
    split = int(np.floor((valid_size+test_size) * num_train))
    np.random.shuffle(indices)
    from torch.utils.data.sampler import SubsetRandomSampler
    
    train_idx, val_test_idx = indices[split:], indices[:split]
    split = int(np.floor(len(val_test_idx)/2))
    val_idx, test_idx = val_test_idx[:split], val_test_idx[split:]
    
    
    train_sampler = SubsetRandomSampler(train_idx)
    val_sampler = SubsetRandomSampler(val_idx)
    test_sampler = SubsetRandomSampler(test_idx)
    
    
    trainloader = torch.utils.data.DataLoader(train_data,
                   sampler=train_sampler, batch_size=size)
    valloader = torch.utils.data.DataLoader(val_data,
                   sampler=val_sampler, batch_size=size)
    testloader = torch.utils.data.DataLoader(test_data,
                   sampler=test_sampler, batch_size=size)
    return trainloader, valloader,testloader

#%%
    
trainloader, valloader, testloader = load_split_train_test(data_dir)
criterion = nn.NLLLoss()

#%%
if model_use == 'resnet50':
    model = models.resnet50(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False
    model.fc = nn.Sequential(nn.Linear(2048, 512),
                                     nn.ReLU(),
                                     nn.Dropout(0.2),
                                     nn.Linear(512, 8),
                                     nn.LogSoftmax(dim=1))
    optimizer = optim.Adam(model.fc.parameters(), lr=0.001)

elif model_use == 'vgg16':
    model = models.vgg16(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False
    model.classifier = nn.Sequential(nn.Linear(25088, 512),
                                     nn.ReLU(),
                                     nn.Dropout(0.2),
                                     nn.Linear(512, 8),
                                     nn.LogSoftmax(dim=1))
    optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)

elif model_use == 'vgg19':
    model = models.vgg19(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False
    model.classifier = nn.Sequential(nn.Linear(25088, 512),
                                     nn.ReLU(),
                                     nn.Dropout(0.2),
                                     nn.Linear(512, 8),
                                     nn.LogSoftmax(dim=1))
    optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)
    
else:
    print('not assign model, defalut vgg16')
    model = models.vgg16(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False
    model.classifier = nn.Sequential(nn.Linear(25088, 512),
                                     nn.ReLU(),
                                     nn.Dropout(0.2),
                                     nn.Linear(512, 8),
                                     nn.LogSoftmax(dim=1))
    optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)

print("model: "+ model_use)
model.to(device)

#%%
epochs = 30
steps = 0
running_loss = 0
print_every = 10
train_losses, test_losses, val_losses = [], [], []
for epoch in range(epochs):
    for inputs, labels in trainloader:
        steps += 1
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        logps = model.forward(inputs)
        loss = criterion(logps, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        
        if steps % print_every == 0:
            test_loss = 0
            test_accuracy = 0
            model.eval()
            with torch.no_grad():
                for inputs, labels in testloader:
                    inputs, labels = inputs.to(device),labels.to(device)
                    logps = model.forward(inputs)
                    batch_loss = criterion(logps, labels)
                    test_loss += batch_loss.item()
                    
                    ps = torch.exp(logps)
                    top_p, top_class = ps.topk(1, dim=1)
                    equals = (top_class == labels.view(*top_class.shape))
                    test_accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
         
            val_loss = 0
            val_accuracy = 0
            with torch.no_grad():
                for inputs, labels in valloader:
                    inputs, labels = inputs.to(device),labels.to(device)
                    logps = model.forward(inputs)
                    batch_loss = criterion(logps, labels)
                    val_loss += batch_loss.item()
                    
                    ps = torch.exp(logps)
                    top_p, top_class = ps.topk(1, dim=1)
                    equals = (top_class == labels.view(*top_class.shape))
                    val_accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
            train_losses.append(running_loss/len(trainloader))
            test_losses.append(test_loss/len(testloader))  
            val_losses.append(val_loss/len(valloader))  

            
            print(f"Epoch {epoch+1}/{epochs}.. "
                  f"Train loss: {running_loss/print_every:.3f}.. "
                  f"Val loss: {val_loss/len(valloader):.3f}.. "
                  f"Test loss: {test_loss/len(testloader):.3f}.. "
                  f"Val accuracy: {val_accuracy/len(valloader):.3f}"
                  f"\nTest accuracy: {test_accuracy/len(testloader):.3f}")
            running_loss = 0
            model.train()

plt.plot(train_losses, label='Training loss')
plt.plot(val_losses, label='Validation loss')
plt.legend(frameon=False)
plt.show()

#torch.save(model, 'model.pth')



















