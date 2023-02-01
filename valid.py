# -*- coding: utf-8 -*-
"""
Created on Fri Jan  6 21:57:19 2023

@author: wnsgm
"""
from sEMG_data import sEMG_data_load as sEMG
from matplotlib import pyplot as plt
import torch
from torch.utils.data import DataLoader, Dataset
from torch import nn
import random

#gpu설정
USE_CUDA=torch.cuda.is_available()
device=torch.device("cuda" if USE_CUDA else 'cpu')
print("다음 기기로 학습합니다.", device)

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
## statics로 train, test set 분류하기

train_mav,train_var,train_wl,test_mav,test_var,test_wl=sEMG().trian_test_set(mav, var, wl)
# label 만들기
label_train_mav,label_test_mav=sEMG().make_label(train_mav, test_mav)

# trian,test 데이터 분류
train_set_mav,test_set_mav,label_train_mav,label_test_mav=sEMG().real_dataset(train_mav,test_mav,label_train_mav,label_test_mav)

# 검증 데이터 만들기
valid_mav,valid_mav_label,test_set_mav,label_test_mav=sEMG().validtaion_dataset(test_set_mav, label_test_mav)

# data tensor로 불러오기
class CustomDataset(Dataset): 
  def __init__(self,x_data_in,y_data_in):
    self.x_data = x_data_in
    self.y_data = y_data_in

  # 총 데이터의 개수를 리턴
  def __len__(self): 
    return len(self.x_data)

  # 인덱스를 입력받아 그에 맵핑되는 입출력 데이터를 파이토치의 Tensor 형태로 리턴
  def __getitem__(self,idx):
    temp=self.x_data[idx]
    x = torch.FloatTensor(temp[:])*100
    y=int(self.y_data[idx])
    return x, y

# 데이터 불러오기
validsets =CustomDataset(valid_mav*100,valid_mav_label)
vaild_loader= DataLoader(validsets, batch_size=5, shuffle=False)

# 학습 모델
class NeuralNet(nn.Module):
    def __init__(self):
        super(NeuralNet,self).__init__()
        
        self.layer_1 = nn.Linear(12, 516, bias=True)
        self.layer_2 = nn.Linear(516, 128, bias=True)
        self.layer_3 = nn.Linear(128, 64, bias=True)
        self.layer_out = nn.Linear(64, 17, bias=True)
        self.dropout = nn.Dropout(0.2)
        self.relu = nn.ReLU()
        self.tanh=nn.Tanh()

        
    def forward(self, x):
        x = self.layer_1(x)
        x = self.tanh(x)
        
        x = self.layer_2(x)
        x = self.tanh(x)
        x = self.dropout(x)
        
        x = self.layer_3(x)
        x = self.tanh(x)
        x = self.dropout(x)
        
        x = self.layer_out(x)
        
        return x

PATH="C:/Users/wnsgm/Desktop/sEMG_final/weight/mav700.pth"
net2=NeuralNet()
net2.load_state_dict(torch.load(PATH))
 
correct=0
total=0

with torch.no_grad():
    net2.eval()
    for data in vaild_loader:
        inputs, labels= data
        outputs=net2(inputs)
        _,predicted=torch.max(outputs.data,1)
        total+=labels.size(0)
        correct+=(predicted==labels).sum().item()
        
print(100*correct/total)
