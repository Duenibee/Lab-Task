# -*- coding: utf-8 -*-
"""
Created on Wed Feb  8 10:28:02 2023

@author: wnsgm
"""

from sEMG_data import sEMG_data_load as sEMG
from matplotlib import pyplot as plt
import torch
from torch.utils.data import DataLoader, Dataset,random_split
from torch import nn
import random
import numpy as np

#gpu설정
USE_CUDA=torch.cuda.is_available()
device=torch.device("cuda" if USE_CUDA else 'cpu')
print("다음 기기로 학습합니다:", device)

random.seed(777)
torch.manual_seed(777)
if device=='cuda':
    torch.cuda.manual_seed_all(777)
 
# data load
x=sEMG().data_load()

# Segmentation
segement1,index=sEMG().segmentation(x)
# mav , var ,wl
mav,var,wl= sEMG().statics_all(segement1,index)
# statics로 train, test set 분류하기

train_mav,train_var,train_wl,test_mav,test_var,test_wl=sEMG().trian_test_set(mav, var, wl)
# label 만들기
label_train_var,label_test_var=sEMG().make_label(train_var, test_var)
label_train_mav,label_test_mav=sEMG().make_label(train_mav, test_mav)
label_train_wl,label_test_wl=sEMG().make_label(train_wl, test_wl)
# trian,test 데이터 분류
train_set_mav,test_set_mav,label_train_mav,label_test_mav=sEMG().real_dataset(train_mav,test_mav,label_train_mav,label_test_mav)
train_set_var,test_set_var,label_train_var,label_test_var=sEMG().real_dataset(train_var,test_var,label_train_var,label_test_var)
train_set_wl,test_set_wl,label_train_wl,label_test_wl=sEMG().real_dataset(train_wl,test_wl,label_train_wl,label_test_wl)

# # valid_data load
# valid_mav,valid_mav_label,test_set_mav,label_test_mav=sEMG().validtaion_dataset(test_set_mav, label_test_mav)
# valid_var,valid_var_label,test_set_var,label_test_var=sEMG().validtaion_dataset(test_set_var, label_test_var)
# valid_wl,valid_wl_label,test_set_wl,label_test_wl=sEMG().validtaion_dataset(test_set_wl, label_test_wl)


# data tensor로 불러오기
class CustomDataset(Dataset): 
  def __init__(self,x_data_mav,y_data_mav,x_data_var,x_data_wl):
    self.x_mav = x_data_mav
    self.y_mav= y_data_mav
    
    self.x_var = x_data_var

    self.x_wl = x_data_wl


  # 총 데이터의 개수를 리턴
  def __len__(self): 
    return len(self.x_mav)

  # 인덱스를 입력받아 그에 맵핑되는 입출력 데이터를 파이토치의 Tensor 형태로 리턴
  def __getitem__(self,idx):
    x=torch.FloatTensor(np.concatenate([self.x_mav[idx],self.x_var[idx],self.x_wl[idx]],axis=0))
    y=int(self.y_mav[idx])
    return x, y


# train 데이터 불러오기
trainsets =CustomDataset(train_set_mav,label_train_mav,train_set_var,train_set_wl)

train_size=int(len(trainsets)*0.8)
valid_size=int(len(trainsets)*0.2)

train_set,valid_set=random_split(trainsets,[train_size, valid_size])
# # valid 데이터 불러오기
# validsets =CustomDataset(valid_mav,valid_mav_label,valid_var,valid_wl)
train_loader= DataLoader(train_set, batch_size=10, shuffle=True)
valid_loader= DataLoader(valid_set, batch_size=1, shuffle=False)

# 학습 모델
class NeuralNet(nn.Module):
    def __init__(self):
        super(NeuralNet,self).__init__()
        self.bn1=nn.BatchNorm1d(128)
        self.bn2=nn.BatchNorm1d(128)
        self.bn3=nn.BatchNorm1d(64)

        self.layer_1 = nn.Linear(36, 128, bias=True)
        self.layer_2 = nn.Linear(128, 128, bias=True)
        self.layer_3 = nn.Linear(128, 64, bias=True)
        self.layer_out = nn.Linear(64, 17, bias=True)
        self.dropout = nn.Dropout(0.2)
        self.tanh=nn.Tanh()
        
    def forward(self, x):
        x = self.layer_1(x)
        # x= self.bn1(x)
        x = self.tanh(x)
        
        
        x = self.layer_2(x)
        # x= self.bn2(x)
        x = self.tanh(x)
        x = self.dropout(x)
        
        x = self.layer_3(x)
        # x= self.bn3(x)
        x = self.tanh(x)
        x = self.dropout(x)

        x = self.layer_out(x)
        
        return x
def saveModel(): 
    path = "./weight/pause_mav_model.pth" 
    torch.save(net.state_dict(), path) 
    
# 모델생성
net= NeuralNet().to(device)
criterion=nn.CrossEntropyLoss().cuda()
#optimizer= optim.SGD(net.parameters(),lr=0.001,momentum=0.9)
optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
def func(epoch):    
    if epoch < 100:
        return 0.5
    elif epoch < 200:
        return 0.5 ** 2
    else:
        return 0.5 ** 3
#scheduler
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda = func)

n=len(train_loader)
n_val=len(valid_loader)
best_accuracy=90
loss_=[]
val_loss_list=[]
for epoch in range(1,300+1):
    net.train()
    correct=0
    total=0
    running_loss=0.0
    for data in train_loader:
        inputs,labels=data
        labels=labels.to(device)
        inputs=inputs.to(device)
        # zero_grad 중복계산을 막기위해 0으로 명시적으로 재설정
        optimizer.zero_grad()
        outputs=net(inputs)
        loss=criterion(outputs.squeeze(),labels.squeeze())
        # loss.backwards() 호출하여 예측 손실(prediction loss)을 역전파
        # optimizer.step()을 호출하여 역전파 단계에서 수집된 변화도로 매개변수를 조정
        loss.backward()
        optimizer.step()
        _,predicted=torch.max(outputs.data,1)
        total+=outputs.size(0)
        correct+=(predicted==labels).sum().item()
        running_loss+=loss.item()
    loss_.append(running_loss/n)  
    scheduler.step()
    # validation
    with torch.no_grad():
        net.eval()
        val_epoch_loss = 0
        val_correct=0
        val_total=0
        for data in valid_loader:
            inputs_val,labels_val=data
            labels_val=labels_val.to(device)
            inputs_val=inputs_val.to(device)
            outputs_val=net(inputs_val)
            val_loss = criterion(outputs_val.squeeze(), labels_val.squeeze())
            _,predicted_val=torch.max(outputs_val.data,1)
            val_total+=outputs_val.size(0)
            val_correct+=(predicted_val==labels_val).sum().item()
            val_epoch_loss+=val_loss.item()
    if 100*val_correct/val_total > best_accuracy: 
        saveModel() 
        best_accuracy = 100*val_correct/val_total
        print("stop training early")
        break
    if epoch%10==0:
        print("\n")
        print(f'--------- Epoch {epoch} ----------')
        print("train_acuuracy:",100*correct/total)
        print("train_loss:",running_loss/n)       
        print("---------------------------")
        print("valid_acuuracy:",100*val_correct/val_total)
        print("valid_loss:",val_epoch_loss/n_val)  
        
print("finished learning")
PATH="C:/Users/wnsgm/Desktop/sEMG_final/weight/all_300.pth"
torch.save(net.state_dict(),PATH)
plt.plot(loss_)
plt.title('MAV_Loss')
plt.xlabel('epoch')
plt.show()


