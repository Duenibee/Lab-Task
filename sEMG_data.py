# -*- coding: utf-8 -*-
"""
Created on Fri Jan 13 16:25:38 2023

@author: wnsgm
"""

import numpy as np
import scipy

mat_file_name="C:/Users/wnsgm/Desktop/sEMG/sEMG 5.mat"
mat_file= scipy.io.loadmat(mat_file_name)

emg=mat_file['emg']
label=mat_file['label']
rep=mat_file['repetition']

emg_np=np.array(emg)
label_np=np.array(label)
rep_np=np.array(rep)



class sEMG_data_load: 
    def __init__(self):
        self.dic1=['L1_1','L1_2','L1_3','L1_4','L1_5','L1_6',
              'L2_1','L2_2','L2_3','L2_4','L2_5','L2_6',
              'L3_1','L3_2','L3_3','L3_4','L3_5','L3_6',
              'L4_1','L4_2','L4_3','L4_4','L4_5','L4_6',
              'L5_1','L5_2','L5_3','L5_4','L5_5','L5_6',
              'L6_1','L6_2','L6_3','L6_4','L6_5','L6_6',
              'L7_1','L7_2','L7_3','L7_4','L7_5','L7_6',
              'L8_1','L8_2','L8_3','L8_4','L8_5','L8_6',
              'L9_1','L9_2','L9_3','L9_4','L9_5','L9_6',
              'L10_1','L10_2','L10_3','L10_4','L10_5','L10_6',
              'L11_1','L11_2','L11_3','L11_4','L11_5','L11_6',
              'L12_1','L12_2','L12_3','L12_4','L12_5','L12_6',
              'L13_1','L13_2','L13_3','L13_4','L13_5','L13_6',
              'L14_1','L14_2','L14_3','L14_4','L14_5','L14_6',
              'L15_1','L15_2','L15_3','L15_4','L15_5','L15_6',
              'L16_1','L16_2','L16_3','L16_4','L16_5','L16_6',
              'L17_1','L17_2','L17_3','L17_4','L17_5','L17_6',]
        
        self.dic2=['L1','L2','L3','L4','L5','L6','L7','L8',
              'L9','L10','L11','L12','L13','L14','L15',
              'L16','L17']

    def data_load(self):
        # label 값 중 0이상의 값만 추출
        label_test=np.where(label_np>0)
        new_arr=label_test[0]
       
        # 만약 new_arr의 값중 연속적이지 않게 증가한 index들을 찾아서 index_list에 넣는다.
        # -> 그때의 index 양끝값이 하나의 덩어리가 된다. 
        index_list=[]
        for i in range(new_arr.size-1):
            if new_arr[i]+1!=new_arr[i+1]:
                index_list.append(new_arr[i])
                index_list.append(new_arr[i+1])
        # index의 시작값 ,끝값 추가
        index_list.insert(0,6899) # 맨앞에 6899라는 index갑 추가
        index_list.append(1802089)
        
        # x 딕셔너리 생성-> 'L1_1', 'L1_2',,,,의 key값으로 emg값들을 나눠서 저장해준다.
        x = {key: value for key, value in dict.fromkeys(self.dic1).items()}
        i=0
        for j in x.keys():
            x[j]=emg_np[index_list[i]:index_list[i+1]+1]
            i+=2
        return x
    
    # segmentation
    # 200ms씩 segementation -> 200ms=0.2sec
    def segmentation(self, x):
        self.x=x
       
        a=[]
        for j in self.dic1:
            # 데이터가 2000hz로 sampling 했기때문에 해당되는 size에 0.0005를 곱해서 계산.
            # 미리 몇개씩 묶을건지 구해준다.
            a.append((x[j].shape[0]*0.0005)//(0.2)) # 나머지는 버릴거기때문에 '//'로 계산
                
        # 각각의 label에 해당되는 segementation들을 segmentation1 딕셔너리에에 저장한다.
        k=0
        segement1 = {key: value for key, value in dict.fromkeys(self.dic1).items()}
        for i in a:
            index_dic=self.dic1[k]
            temp=x[index_dic]
            segement1[index_dic]=temp[0:400*int(i)] # 200ms=0.2s, 0.2*2000hz=400samples 
            k+=1
        index=a
        return segement1,index
    
    # 통계적 수치구하기 
    def statics_all(self,segement1,index):
        # 딕셔너리 생성
        statics_mav={}
        statics_mav = {key: value for key, value in dict.fromkeys(self.dic1).items()}
        
        statics_var={}
        statics_var = {key: value for key, value in dict.fromkeys(self.dic1).items()}
        
        statics_wl={}
        statics_wl = {key: value for key, value in dict.fromkeys(self.dic1).items()}
       
        k=0
        for i in index:
            index_dic=self.dic1[k]
            temp=segement1[index_dic]
            # 첫번째 값들 넣어놓기-> np.concatenate를 사용하기위해
            # mav
            mav=np.sum(np.absolute(temp[0:400]),axis=0,keepdims=True)/400
            # var
            var=np.sum(temp[0:400]**2,axis=0,keepdims=True)/(400-1)
            # wl
            temp_wl=0
            for p in range(0,400-1):
                temp_wl+=np.absolute(temp[p+1]-temp[p])
            
            # 차원 맞춰주기 mav, var은 keepdims=True여서 해줄 필요x
            wl=np.reshape(temp_wl,(1,12))
            statics_mav[index_dic]=np.array(mav)
            statics_var[index_dic]=np.array(var)
            statics_wl[index_dic]=np.array(wl)
            
            # 1번째부터 통계적 수치 구하기 
            for j in range(1,int(i)):        
                # mav
                mav=np.sum(np.absolute(temp[400*j:400*j+400]),keepdims=True,axis=0)/400
                # var
                var=np.sum(temp[400*j:400*j+400]**2,keepdims=True,axis=0)/(400-1)
                # wl
                temp_wl=0
                for p in range(400*j,400*j+400-1):
                    temp_wl+=np.absolute(temp[p+1]-temp[p])
                wl=np.reshape(temp_wl,(1,12))
                # 계산결과를 np.concatenate해줘서 하나의 데이터로 만들어준다.
                statics_mav[index_dic]=np.concatenate((statics_mav[index_dic],mav),axis=0) 
                statics_var[index_dic]=np.concatenate((statics_var[index_dic],var),axis=0)
                statics_wl[index_dic]=np.concatenate((statics_wl[index_dic], wl),axis=0) 
            k+=1
        return statics_mav,statics_var,statics_wl
    
    def trian_test_set(self,mav,var,wl):
        dic2=self.dic2
        dic1=self.dic1
        statics_mav=mav
        statics_var=var
        statics_wl=wl
        
        train_set_mav = {key: value for key, value in dict.fromkeys(dic2).items()}
        test_set_mav ={key: value for key, value in dict.fromkeys(dic2).items()}

        train_set_var= {key: value for key, value in dict.fromkeys(dic2).items()}
        test_set_var={key: value for key, value in dict.fromkeys(dic2).items()}

        train_set_wl = {key: value for key, value in dict.fromkeys(dic2).items()}
        test_set_wl ={key: value for key, value in dict.fromkeys(dic2).items()}
        
        i=0
        while(i<len(dic1)):
            index_dic1=dic1[i]
            index_dic2=dic2[i//6]  
            if i%6==0: # 'L1_1'처럼 _1들을 첫번째 train 데이터로 넣어준다.
                train_set_mav[index_dic2]=statics_mav[index_dic1]
                train_set_var[index_dic2]=statics_var[index_dic1]
                train_set_wl[index_dic2]=statics_wl[index_dic1]
            elif ((i%6)<4): # 0,1,2,3 의 인덱스들은 -> train_set
                train_set_mav[index_dic2]=np.concatenate((train_set_mav[index_dic2],statics_mav[index_dic1]),axis=0)
                train_set_var[index_dic2]=np.concatenate((train_set_var[index_dic2],statics_var[index_dic1]),axis=0)
                train_set_wl[index_dic2]=np.concatenate((train_set_wl[index_dic2],statics_wl[index_dic1]),axis=0)
            elif ((i%6)==4):  #'L1_4'처럼 _4들을 첫번째 test 데이터로 넣어준다.
                test_set_mav[index_dic2]=statics_mav[index_dic1]
                test_set_var[index_dic2]=statics_var[index_dic1]
                test_set_wl[index_dic2]=statics_wl[index_dic1]
            else: # 나머지는다 test set
                test_set_mav[index_dic2]=np.concatenate((test_set_mav[index_dic2],statics_mav[index_dic1]),axis=0)
                test_set_var[index_dic2]=np.concatenate((test_set_var[index_dic2],statics_var[index_dic1]),axis=0)
                test_set_wl[index_dic2]=np.concatenate((test_set_wl[index_dic2],statics_wl[index_dic1]),axis=0)
            i+=1     
        
        return train_set_mav,train_set_var,train_set_wl,test_set_mav,test_set_var,test_set_wl
    
    # label 값 만들기
    def make_label(self,train_set,test_set):
        dic2=self.dic2
        label_train={key: value for key, value in dict.fromkeys(dic2).items()}
        label_test={key: value for key, value in dict.fromkeys(dic2).items()}

        for i in range(0,len(dic2)):
            index_dic2=dic2[i]
            label_train[index_dic2]=np.full(((train_set[index_dic2].shape)[0],1),i)
            label_test[index_dic2]=np.full(((test_set[index_dic2].shape)[0],1),i)
        return label_train,label_test
    
    
    def real_dataset(self,train_set,test_set,label_train,label_test):
        dic2=self.dic2 
        for i in range(0,len(dic2)):
            index_dic2=dic2[i]
            
            temp_train=train_set[index_dic2]
            temp_test=test_set[index_dic2]
            
            temp_label_train=label_train[index_dic2]
            temp_label_test=label_test[index_dic2]
            
            label_train[index_dic2]=temp_label_train[0:50]
            label_test[index_dic2]=temp_label_test[0:30]
            
            train_set[index_dic2]=temp_train[0:50]
            test_set[index_dic2]=temp_test[0:30]
        
        # 데이터 한 덩어리로 만들기
        real_train=train_set['L1']
        real_test=test_set['L1']

        real_label_train=label_train['L1']
        real_label_test=label_test['L1']
        
        for i in range(1,len(dic2)):
            index_dic2=dic2[i]
            real_train=np.concatenate((real_train,train_set[index_dic2]),axis=0)
            real_test=np.concatenate((real_test,test_set[index_dic2]),axis=0)
            
            real_label_train=np.concatenate((real_label_train,label_train[index_dic2]),axis=0)
            real_label_test=np.concatenate((real_label_test,label_test[index_dic2]),axis=0)
            
        return real_train,real_test,real_label_train,real_label_test
    
    def validtaion_dataset(self,test_set,test_label):
        num=510
        valid_set=test_set[list(range(0,num,2))][:]
        valid_label_set=test_label[list(range(0,num,2))][:]
        test_set=test_set[list(range(1,num,2))][:]
        test_label=test_label[list(range(1,num,2))][:]
        return valid_set,valid_label_set,test_set,test_label