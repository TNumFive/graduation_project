import pandas as pd
import numpy as np
import sys
import os
np.random.seed(seed=5)

def prepare_data_tt(forestep=32):
    file_list=os.listdir('./data')
    if 'tt_dataset_multi_output.csv' in file_list:
        dataset:pd.DataFrame=pd.read_csv('./data/tt_dataset_multi_output.csv',parse_dates=['start_time'])
    else:
        dataset=pd.DataFrame()
        ttdataset=pd.read_csv('./data/tt_dataset.csv',parse_dates=['start_time'])
        ttdataset.set_index(['start_time'],inplace=True,drop=False)
        len_ttdataset=len(ttdataset)
        for i in range(forestep+1,len_ttdataset):
            start:pd.Timestamp=ttdataset.iloc[i-forestep-1].name
            now:pd.Timestamp=ttdataset.iloc[i].name
            if now.day-start.day<=1:
                dataset=dataset.append(ttdataset.iloc[i-forestep-1:i],ignore_index=True)
            print('\r\tprepare data: ',i+1,'/',len_ttdataset,sep='',end='')
        print('')
        dataset.to_csv('./data/tt_dataset_multi_output.csv',index=False)
    dataset.set_index(['start_time'],inplace=True)
    dataset=dataset.to_numpy().reshape(int(len(dataset)/(forestep+1)),forestep+1,len(dataset.columns))
    dataset=dataset/60#turn seconds to mins to decrease calculation
    print('dataset.shape:',dataset.shape)
    splitratio=int(0.7*dataset.shape[0])
    train=dataset[:splitratio]
    x_train=train[:,:-1]
    y_train=train[:,-1]
    test=dataset[splitratio:]
    x_test=test[:,:-1]
    y_test=test[:,-1]
    print('x_train.shape:',x_train.shape)
    print('y_train.shape:',y_train.shape)
    return x_train,x_test,y_train,y_test

def prepare_data_pf(forestep=32):
    file_list=os.listdir('./data')
    if 'pf_dataset_multi_output.csv' in file_list:
        dataset:pd.DataFrame=pd.read_csv('./data/pf_dataset_multi_output.csv',parse_dates=['start_time'])
    else:
        dataset=pd.DataFrame()
        pfdataset=pd.read_csv('./data/pf_dataset.csv',parse_dates=['start_time'])
        pfdataset.set_index(['start_time'],inplace=True,drop=False)
        len_pfdataset=len(pfdataset)
        for i in range(forestep+1,len_pfdataset):
            start:pd.Timestamp=pfdataset.iloc[i-forestep-1].name
            now:pd.Timestamp=pfdataset.iloc[i].name
            if now.day-start.day<=1:
                dataset=dataset.append(pfdataset.iloc[i-forestep-1:i],ignore_index=True)
            print('\r\tprepare data: ',i+1,'/',len_pfdataset,sep='',end='')
        print('')
        dataset.to_csv('./data/pf_dataset_multi_output.csv',index=False)
    dataset.set_index(['start_time'],inplace=True)
    dataset=dataset.to_numpy().reshape(int(len(dataset)/(forestep+1)),forestep+1,len(dataset.columns))
    
    print('dataset.shape:',dataset.shape)
    splitratio=int(0.7*dataset.shape[0])
    train=dataset[:splitratio]
    x_train=train[:,:-1]
    y_train=train[:,-1]
    test=dataset[splitratio:]
    x_test=test[:,:-1]
    y_test=test[:,-1]
    print('x_train.shape:',x_train.shape)
    print('y_train.shape:',y_train.shape)
    return x_train,x_test,y_train,y_test

def prepare_data_ttpf(forestep=8,foreday=3,slot_length=15):
    from sklearn.preprocessing import MinMaxScaler

    ttdataset=pd.read_csv('./data/tt_dataset.csv',parse_dates=['start_time'])
    ttdataset.set_index(['start_time'],inplace=True)
    pfdataset=pd.read_csv('./data/pf_dataset.csv',parse_dates=['start_time'])
    pfdataset.set_index(['start_time'],inplace=True)
    
    len_dataset=len(ttdataset)
    splitpoint=int(0.7*len_dataset)
    
    ttmms=MinMaxScaler()
    tt_train=ttdataset.to_numpy()[:splitpoint].flatten().reshape(-1,1)
    tt_test=ttdataset.to_numpy()[splitpoint:].flatten().reshape(-1,1)
    tt_train=ttmms.fit_transform(tt_train)
    tt_test=ttmms.transform(tt_test)
    tt_train=tt_train.reshape((int(len(tt_train)/20),20,1))
    tt_test=tt_test.reshape((int(len(tt_test)/20),20,1))

    pfmms=MinMaxScaler()
    pf_train=pfdataset.to_numpy()[:splitpoint].flatten().reshape(-1,1)
    pf_test=pfdataset.to_numpy()[splitpoint:].flatten().reshape(-1,1)
    pf_train=pfmms.fit_transform(pf_train)
    pf_test=pfmms.transform(pf_test)
    pf_train=pf_train.reshape((int(len(pf_train)/20),20,1))
    pf_test=pf_test.reshape((int(len(pf_test)/20),20,1))
    
    print(pf_train.shape,pf_test.shape,tt_train.shape,tt_test.shape)
    train=np.concatenate((tt_train,pf_train),axis=-1)
    print(train.shape)


def prepare_data_mm(filepath:str):
    from datetime import datetime
    rawdata=pd.read_csv(filepath,parse_dates=['start_time'])
    rawdata.set_index(['start_time'],inplace=True,drop=False)
    start_time=rawdata.loc[:,'start_time']
    dataset=pd.DataFrame()
    clock=datetime.now()
    for st in start_time:
        for day in range(-3,0):
            if st+pd.Timedelta(days=day) not in start_time:
                st=0
                break
        if st!=0:
            for day in range(-3,0):
                part=pd.DataFrame()
                for i in range(-4,0):
                    temp:pd.Timestamp=st+pd.Timedelta(days=day,minutes=15*i)
                    if temp in start_time:
                        part=part.append(rawdata.loc[temp,:],ignore_index=True)
                    else:
                        part=part.append({'start_time':temp},ignore_index=True)
                for j in range(0,5):
                    temp:pd.Timestamp=st+pd.Timedelta(days=day,minutes=15*j)
                    if temp in start_time:
                        part=part.append(rawdata.loc[temp,:],ignore_index=True)
                    else:
                        part=part.append({'start_time':temp},ignore_index=True)
                part.fillna(method='bfill',inplace=True)
                part.fillna(method='ffill',inplace=True)
                dataset=dataset.append(part,ignore_index=True)
            part=pd.DataFrame()
            for k in range(-8,1):
                temp:pd.Timestamp=st+pd.Timedelta(minutes=15*k)
                if temp in start_time:
                    part=part.append(rawdata.loc[temp,:],ignore_index=True)
                else:
                    part=part.append({'start_time':temp},ignore_index=True)
            part.fillna(value=0,inplace=True)
            for l in range(1,2):
                temp:pd.Timestamp=st+pd.Timedelta(minutes=15*l)
                if temp in start_time:
                    part=part.append(rawdata.loc[temp,:],ignore_index=True)
                else:
                    part=part.append({'start_time':temp},ignore_index=True)
            part.fillna(method='ffill',inplace=True)
            dataset=dataset.append(part,ignore_index=True)
        print('\r\t',st,end='')
    print('')
    print(datetime.now()-clock)
    dataset.to_csv('./data/test.csv',index=False)

if __name__ == "__main__":
    #prepare_data_ttpf()    
    prepare_data_mm('./data/tt_dataset.csv')