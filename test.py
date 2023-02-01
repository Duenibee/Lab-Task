from sEMG_data import sEMG_data_load as sEMG
import torch
from torch.utils.data import DataLoader, Dataset
from torch import nn
import random
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
# gpu 설정
USE_CUDA=torch.cuda.is_available()
device=torch.device("cuda" if USE_CUDA else 'cpu')
print("다음 기기로 학습합니다.", device)

random.seed(777)
torch.manual_seed(777)
if device=='cuda':
    torch.cuda.manual_seed_all(777)
 
#data load
x=sEMG().data_load()

# Segmentation
segement1,index=sEMG().segmentation(x)
# mav , var ,wl
mav,var,wl= sEMG().statics_all(segement1,index)
## statics로 train, test set 분류하기

train_mav,train_var,train_wl,test_mav,test_var,test_wl=sEMG().trian_test_set(mav, var, wl)
# label 만들기
label_train_mav,label_test_mav=sEMG().make_label(train_mav, test_mav)
label_train_var,label_test_var=sEMG().make_label(train_var, test_var)
label_train_wl,label_test_wl=sEMG().make_label(train_wl, test_wl)
# trian,test 데이터 분류
train_set_mav,test_set_mav,label_train_mav,label_test_mav=sEMG().real_dataset(train_mav,test_mav,label_train_mav,label_test_mav)
train_set_var,test_set_var,label_train_var,label_test_var=sEMG().real_dataset(train_var,test_var,label_train_var,label_test_var)
train_set_wl,test_set_wl,label_train_wl,label_test_wl=sEMG().real_dataset(train_wl,test_wl,label_train_wl,label_test_wl)

valid_mav,valid_mav_label,test_set_mav,label_test_mav=sEMG().validtaion_dataset(test_set_mav, label_test_mav)
valid_var,valid_var_label,test_set_var,label_test_var=sEMG().validtaion_dataset(test_set_var, label_test_var)
valid_wl,valid_wl_label,test_set_wl,label_test_wl=sEMG().validtaion_dataset(test_set_wl, label_test_wl)
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

testsets= CustomDataset(test_set_var, label_test_var)
test_loader= DataLoader(testsets, batch_size=1, shuffle=False)

class2idx = {
        "L1": 0,
        "L2": 1,
        "L3": 2,
        "L4": 3,
        "L5": 4,
        "L6": 5,
        "L7": 6,
        "L8": 7,
        "L9": 8,
        "L10": 9,
        "L11": 10,
        "L12": 11,
        "L13": 12,
        "L14": 13,
        "L15": 14,
        "L16": 15,
        "L17": 16,
}

idx2class = {v: k for k, v in class2idx.items()}


# 학습 모델
class NeuralNet(nn.Module):
    def __init__(self):
        super(NeuralNet,self).__init__()
        self.bn1=nn.BatchNorm1d(128)
        self.bn2=nn.BatchNorm1d(128)
        self.bn3=nn.BatchNorm1d(64)

        self.layer_1 = nn.Linear(12, 128, bias=True)
        self.layer_2 = nn.Linear(128, 128, bias=True)
        self.layer_3= nn.Linear(128, 64, bias=True)
        self.layer_out = nn.Linear(64, 17, bias=True)
        self.dropout = nn.Dropout(0.2)
        self.relu = nn.ReLU()
        self.tanh=nn.Tanh()
        
    def forward(self, x):
        x = self.layer_1(x)
        x = self.tanh(x)
        x= self.bn1(x)
        
        x = self.layer_2(x)
        x = self.tanh(x)
        x = self.dropout(x)
        x= self.bn2(x)
        
        x = self.layer_3(x)
        x = self.tanh(x)
        x = self.dropout(x)
        x= self.bn3(x)
        
        x = self.layer_out(x)
        
        return x

# 해당되는 .pth 파일 적용
PATH="C:/Users/wnsgm/Desktop/sEMG_final/weight/var_700.pth"
net2=NeuralNet()
net2.load_state_dict(torch.load(PATH))

correct=0
total=0
y_pred_list=[]
with torch.no_grad():
    net2.eval()
    for data in test_loader:
        inputs, labels= data
        outputs=net2(inputs)
        _,predicted=torch.max(outputs.data,1)
        total+=labels.size(0)
        correct+=(predicted==labels).sum().item()
        y_pred_list.append(predicted.cpu().numpy())
y_pred_list = [a.squeeze().tolist() for a in y_pred_list]
print(100*correct/total)

# confusion_matrix
confusion_matrix_df = pd.DataFrame(confusion_matrix(label_test_var, y_pred_list)).rename(columns=idx2class, index=idx2class)

sns.heatmap(confusion_matrix_df, annot=True)
print(classification_report(label_test_var, y_pred_list))