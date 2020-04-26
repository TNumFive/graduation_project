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

def prepare_data_ttpf(slot_length=15):
    from sklearn.preprocessing import MinMaxScaler

    ttdataset=pd.read_csv('./data/tt_dataset_mymodel.csv',parse_dates=['start_time'])
    pfdataset=pd.read_csv('./data/pf_dataset_mymodel.csv',parse_dates=['start_time'])

    index_tt=ttdataset.pop('start_time')
    index_pf=pfdataset.pop('start_time')
    weekday=pd.DataFrame()
    for i in range(0,len(index_tt)):
        time=index_tt[i]
        weekday.at[i,'weekday']=time.weekday()
        print('\r\t',i,'/',len(index_tt),end='')
    print('')
    weekday=weekday.to_numpy().reshape((int(len(ttdataset)/37),37,1))
    ttdataset=ttdataset.to_numpy().reshape((int(len(ttdataset)/37),37,20))
    pfdataset=pfdataset.to_numpy().reshape((int(len(pfdataset)/37),37,20))
    split=int(ttdataset.shape[0]*0.7)
    
    ttmms=MinMaxScaler()
    tt_train=ttdataset[:split,:].flatten().reshape((-1,1))
    tt_test=ttdataset[split:,:].flatten().reshape((-1,1))
    tt_train=ttmms.fit_transform(tt_train)
    tt_test=ttmms.transform(tt_test)
    tt_train=tt_train.reshape((split,37,20))
    tt_test=tt_test.reshape((len(ttdataset)-split,37,20))

    pfmms=MinMaxScaler()
    pf_train=ttdataset[:split,:].flatten().reshape((-1,1))
    pf_test=ttdataset[split:,:].flatten().reshape((-1,1))
    pf_train=pfmms.fit_transform(pf_train)
    pf_test=pfmms.transform(pf_test)
    pf_train=pf_train.reshape((split,37,20))
    pf_test=pf_test.reshape((len(ttdataset)-split,37,20))
    
    print(weekday.shape)
    print(index_tt[0])
    print(tt_train.shape)
    print(index_pf[0])
    print(pf_train.shape)
    data={
        'weekday':weekday,
        'tt_train':tt_train,
        'tt_test':tt_test,
        'ttmms':ttmms,
        'pf_train':pf_train,
        'pf_test':pf_test,
        'pfmms':pfmms
    }
    return data

def prepare_data_mm(filepath:str,savepath:str,is_tt=1):
    #预测第d天的t，t+1时隙
    #长期依赖 d-3，d-2，d-1天的 t-4～t+4时隙
    #短期依赖 d天t-8～t-1时隙
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
                if is_tt:
                    part.fillna(method='bfill',inplace=True)
                    part.fillna(method='ffill',inplace=True)
                else:
                    part.fillna(value=0,inplace=True)
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
            if is_tt:
                part.fillna(method='ffill',inplace=True)
            else:
                part.fillna(value=0,inplace=True)
            dataset=dataset.append(part,ignore_index=True)
        print('\r\t',st,end='')
    print('')
    print(datetime.now()-clock)
    dataset.to_csv(savepath,index=False)

if __name__ == "__main__":
    prefix=os.path.basename(sys.argv[0])
    print(prefix)
    #prepare_data_ttpf()    
    #prepare_data_mm('./data/pf_dataset.csv','./data/pf_dataset_mymodel.csv',0)
    prepare_data_ttpf()