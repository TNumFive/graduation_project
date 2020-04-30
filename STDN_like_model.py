import numpy as np
import pandas as pd
import datetime
import os
import sys

from matplotlib import pyplot
from keras import layers
from keras import Model
from keras import models
from keras import backend as K
from keras import callbacks
from keras.utils import plot_model
from sklearn.preprocessing import MinMaxScaler,OneHotEncoder
from sklearn.metrics import mean_absolute_error,mean_squared_error

np.random.seed(seed=5)

def print_log(s:str,end='\n',log_name='runtime.log'):
    runtime=open(log_name,'a+')
    runtime.write(s+end)
    runtime.close()

class attention_layer(layers.Layer):#simplest attention layer
    def __init__(self, **kwargs):
        super(attention_layer, self).__init__(**kwargs)

    def build(self,input_shape):#input_shape=(,9,64)
        assert len(input_shape)==3
        self.W=self.add_weight(name='attr_weight',shape=(input_shape[1],input_shape[2]),initializer='uniform',trainable=True)
        self.b=self.add_weight(name='attr_bias',shape=(input_shape[2],),initializer='uniform',trainable=True)
        super(attention_layer,self).build(input_shape)

    def call(self,inputs):
        x=K.permute_dimensions(inputs,(0,2,1))#(,9,64)->(,64,9)
        a=K.dot(x,self.W) #(,64,9).(,9,64)->(64,64)
        a_prob=K.bias_add(a,self.b)#(64,64)
        a=K.tanh(a_prob)#(64,64)
        a=K.softmax(a,axis=1)#(64,64)
        a=a*a_prob
        a=K.permute_dimensions(a,(0,2,1))
        a=K.sum(a,axis=1)
        return a

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[2]

def STDN_like(
    name='STDN_like',
    sta_num=20,
    lt_shape=(3,9,20),
    st_shape=(8,20),
    wd_shape=(4,1),
    op_shape=(1,20),
    optimizer='adam',
    metrics=['mae'],
    loss='mse'
)->Model:
    lt=layers.Input(shape=lt_shape)
    st=layers.Input(shape=st_shape)
    wd=layers.Input(shape=wd_shape)

    y1=layers.Reshape((lt_shape[0]*lt_shape[1],lt_shape[2],1))(lt)
    y1=layers.TimeDistributed(layers.Conv1D(filters=64,kernel_size=10,padding='same'))(y1)
    #y1=layers.Activation('sigmoid')
    y1=layers.Activation('relu')(y1)
    y1=layers.Reshape((lt_shape[0],lt_shape[1],lt_shape[2]*64))(y1)
    y1=layers.TimeDistributed(layers.LSTM(64,return_sequences=True,dropout=0.1,recurrent_dropout=0.1))(y1)
    y1=layers.TimeDistributed(layers.LSTM(64,return_sequences=True,dropout=0.1,recurrent_dropout=0.1))(y1)
    y1=layers.TimeDistributed(attention_layer())(y1)

    y2=layers.Reshape((st_shape[0],st_shape[1],1))(st)
    y2=layers.TimeDistributed(layers.Conv1D(filters=64,kernel_size=6,padding='same'))(y2)
    y2=layers.Activation('relu')(y2)
    y2=layers.Reshape((st_shape[0],st_shape[1]*64))(y2)
    y2=layers.LSTM(64,return_sequences=True,dropout=0.1,recurrent_dropout=0.1)(y2)
    y2=layers.LSTM(64,return_sequences=True,dropout=0.1,recurrent_dropout=0.1)(y2)
    y2=attention_layer()(y2)
    y2=layers.Reshape((1,64))(y2)

    y=layers.concatenate([y1,y2],axis=1)
    y=layers.concatenate([y,wd])
    y=layers.Flatten()(y)
    y=layers.Dense(op_shape[0]*op_shape[1])(y)
    
    model=Model(inputs=[lt,st,wd],outputs=[y],name=name)
    model.compile(optimizer=optimizer,metrics=metrics,loss=loss)

    return model

def STDN_like2(
    name='STDN_like2',
    sta_num=20,
    lt_shape=(3,9,20),
    st_shape=(8,20),
    wd_shape=(4,1),
    op_shape=(1,20),
    optimizer='adam',
    metrics=['mae'],
    loss='mse'
)->Model:
    lt=layers.Input(shape=lt_shape)
    st=layers.Input(shape=st_shape)
    wd=layers.Input(shape=wd_shape)

    y1=layers.Reshape((lt_shape[0]*lt_shape[1],lt_shape[2],1))(lt)
    y1=layers.TimeDistributed(layers.Conv1D(filters=64,kernel_size=10,padding='same'))(y1)
    y1=layers.LeakyReLU()(y1)
    y1=layers.TimeDistributed(layers.Conv1D(filters=64,kernel_size=5,padding='same'))(y1)
    y1=layers.Activation('relu')(y1)
    y1=layers.Activation('sigmoid')(y1)

    y2=layers.Reshape((st_shape[0],st_shape[1],1))(st)
    y2=layers.TimeDistributed(layers.Conv1D(filters=64,kernel_size=10,padding='same'))(y2)
    y2=layers.LeakyReLU()(y2)
    y2=layers.TimeDistributed(layers.Conv1D(filters=64,kernel_size=5,padding='same'))(y2)
    y2=layers.Activation('relu')(y2)
    y2=layers.Activation('sigmoid')(y2)

    y=layers.concatenate([y1,y2],axis=1)
    y=layers.Reshape((lt_shape[0]*lt_shape[1]+st_shape[0],sta_num*64))(y)
    y=layers.LSTM(64,return_sequences=True)(y)
    y=layers.AlphaDropout(0.2)(y)
    y=layers.LSTM(64,return_sequences=True)(y)
    y=layers.AlphaDropout(0.1)(y)
    y=attention_layer()(y)
    y=layers.Dense(op_shape[0]*op_shape[1])(y)
    model=Model(inputs=[lt,st,wd],outputs=[y],name=name)
    model.compile(optimizer=optimizer,metrics=metrics,loss=loss)

    return model

def onehot_external_features(filepath='weather.csv')->(dict,dict):
    ef=pd.read_csv(filepath,parse_dates=['date'])
    enc=OneHotEncoder()
    weather_list=list()
    wd_list=list()#wind direction
    for row in ef.iterrows():
        if row[1]['dw'] not in weather_list:
            weather_list.append(row[1]['dw'])
        if row[1]['nw'] not in weather_list:
            weather_list.append(row[1]['nw'])
        if row[1]['wd'] not in wd_list:
            wd_list.append(row[1]['wd'])
    weather_array=np.array(weather_list).reshape((-1,1))
    wd_array=np.array(wd_list).reshape((-1,1))
    weather_array=enc.fit_transform(weather_array).toarray()
    wd_array=enc.fit_transform(wd_array).toarray()
    weather_dict=dict()
    for i in range(0,len(weather_array)):
        weather_dict[weather_list[i]]=list(weather_array[i])
    wd_dict=dict()
    for i in range(0,len(wd_array)):
        wd_dict[wd_list[i]]=list(wd_array[i])
    return weather_dict,wd_dict

def preprocess_data(
    filepath='tt_dataset.csv',
    savepath='tt_dataset_STDN_like.csv',
    #is_tt=1,
    #sta_num=20,
    slot_length=15,
    lt_shape=(3,9,20),
    st_shape=(8,20),
    wd_shape=(4,1),
    op_shape=(1,20)
)->pd.DataFrame:
    from datetime import datetime
    #预测第d天的t~t+op_shape[0]时隙
    #长期依赖 d-3，d-2，d-1天的 t-4～t+4时隙
    #短期依赖 d天t-8～t-1时隙
    print('preprocess data',filepath)
    assert len(lt_shape)==3
    assert len(st_shape)==2
    assert len(wd_shape)==2
    assert len(op_shape)==2
    file_list=os.listdir('./')
    if savepath in file_list:
        data=pd.read_csv(savepath,parse_dates=['start_time'])
    else:
        clock=datetime.now()
        data=pd.DataFrame()
        raw=pd.read_csv(filepath,parse_dates=['start_time'])
        raw.set_index(['start_time'],inplace=True,drop=False)
        start_time=raw.loc[:,'start_time']
        for st in start_time:
            for day in range(0-lt_shape[0],0):#check if long term dependency exists
                if st+pd.Timedelta(days=day) not in start_time:
                    st=0
                    break
            for op in range(1,op_shape[0]):#check if it has enough slot to output
                if st+pd.Timedelta(minutes=slot_length*op) not in start_time:
                    st=0
                    break
            if st!=0:
                for day in range(0-lt_shape[0],0):
                    part=pd.DataFrame()
                    for i in range(int(0-lt_shape[1]/2),0):
                        temp:pd.Timestamp=st+pd.Timedelta(days=day,minutes=slot_length*i)
                        if temp in start_time:
                            part=part.append(raw.loc[temp,:],ignore_index=True)
                        else:
                            part=part.append({'start_time':temp},ignore_index=True)
                    for j in range(0,int(1+lt_shape[1]/2)):
                        temp:pd.Timestamp=st+pd.Timedelta(days=day,minutes=slot_length*j)
                        if temp in start_time:
                            part=part.append(raw.loc[temp,:],ignore_index=True)
                        else:
                            part=part.append({'start_time':temp},ignore_index=True)
                    '''
                    if is_tt:
                        part.fillna(method='bfill',inplace=True)
                        part.fillna(method='ffill',inplace=True)
                    else:
                        part.fillna(value=0,inplace=True)
                    '''
                    part.fillna(value=0,inplace=True)
                    data=data.append(part,ignore_index=True)
                part=pd.DataFrame()
                for k in range(0-st_shape[0],1):
                    temp:pd.Timestamp=st+pd.Timedelta(minutes=slot_length*k)
                    if temp in start_time:
                        part=part.append(raw.loc[temp,:],ignore_index=True)
                    else:
                        part=part.append({'start_time':temp},ignore_index=True)
                part.fillna(value=0,inplace=True)
                for l in range(1,op_shape[0]):
                    temp:pd.Timestamp=st+pd.Timedelta(minutes=slot_length*l)
                    if temp in start_time:
                        part=part.append(raw.loc[temp,:],ignore_index=True)
                    else:
                        part=part.append({'start_time':temp},ignore_index=True)
                '''
                if is_tt:
                    part.fillna(method='ffill',inplace=True)
                else:
                    part.fillna(value=0,inplace=True)
                '''
                part.fillna(value=0,inplace=True)
                data=data.append(part,ignore_index=True)
            print('\r\tpreprocessing data: ts',st,end='')
        print('')
        data.to_csv(savepath,index=False)
        print('time consumed:',datetime.now()-clock)
    return data

def generate_data1(
    sta_num=20,
    slot_length=15,
    lt_shape=(3,9,20),
    st_shape=(8,20),
    wd_shape=(4,1),
    op_shape=(1,20),
    spilitratio=0.7
):
    print('generate data')
    tt=preprocess_data(filepath='tt_dataset.csv',savepath='tt_dataset_STDN_like.csv')
    pf=preprocess_data(filepath='pf_dataset.csv',savepath='pf_dataset_STDN_like.csv')#thr original only model ttdata
    assert len(tt)==len(pf)

    start_time=tt.pop('start_time')
    tt=tt.to_numpy().reshape((
        int(len(start_time)/(lt_shape[0]*lt_shape[1]+st_shape[0]+op_shape[0])),
        lt_shape[0]*lt_shape[1]+st_shape[0]+op_shape[0],
        sta_num
    ))
    spilitpoint=int(spilitratio*tt.shape[0])
    train_tt=tt[:spilitpoint].reshape((-1,1))
    test_tt=tt[spilitpoint:].reshape((-1,1))
    mms_tt=MinMaxScaler()
    train_tt:np.ndarray=mms_tt.fit_transform(train_tt)
    train_tt=train_tt.reshape((
        spilitpoint,
        lt_shape[0]*lt_shape[1]+st_shape[0]+op_shape[0],
        sta_num
    ))
    test_tt:np.ndarray=mms_tt.transform(test_tt)
    test_tt=test_tt.reshape((
        tt.shape[0]-spilitpoint,
        lt_shape[0]*lt_shape[1]+st_shape[0]+op_shape[0],
        sta_num
    ))
    
    start_time=pf.pop('start_time')
    pf=pf.to_numpy().reshape((
        int(len(start_time)/(lt_shape[0]*lt_shape[1]+st_shape[0]+op_shape[0])),
        lt_shape[0]*lt_shape[1]+st_shape[0]+op_shape[0],
        sta_num
    ))
    spilitpoint=int(spilitratio*pf.shape[0])
    train_pf=pf[:spilitpoint].reshape((-1,1))
    test_pf=pf[spilitpoint:].reshape((-1,1))
    mms_pf=MinMaxScaler()
    train_pf:np.ndarray=mms_pf.fit_transform(train_pf)
    train_pf=train_pf.reshape((
        spilitpoint,
        lt_shape[0]*lt_shape[1]+st_shape[0]+op_shape[0],
        sta_num
    ))
    test_pf:np.ndarray=mms_pf.transform(test_pf)
    test_pf=test_pf.reshape((
        pf.shape[0]-spilitpoint,
        lt_shape[0]*lt_shape[1]+st_shape[0]+op_shape[0],
        sta_num
    ))
    
    print('generate external features')
    weather_dict,wd_dict=onehot_external_features()#
    ef=pd.read_csv('weather.csv',parse_dates=['date'])
    ef.set_index(['date'],inplace=True)
    weekday=list()
    ef_list=list()
    len_start_time=len(start_time)
    for i in range(0,len_start_time,lt_shape[0]*lt_shape[1]+st_shape[0]+op_shape[0]):
        for j in range(0,lt_shape[0]*lt_shape[1]+st_shape[0],9):
            ts:pd.Timestamp=start_time[i+j]
            if ts.weekday()>=5:
                weekday.append(1)
            else:
                weekday.append(0)
            ef_list.append(ef.at[ts.date(),'AQI'])
            ef_list+=weather_dict[ef.at[ts.date(),'dw']]
            ef_list+=weather_dict[ef.at[ts.date(),'nw']]
            ef_list.append(ef.at[ts.date(),'ht'])
            ef_list.append(ef.at[ts.date(),'lt'])
            ef_list+=wd_dict[ef.at[ts.date(),'wd']]
            ef_list.append(ef.at[ts.date(),'wf'])
        print('\r\tdone',i+1,'/',len_start_time,end='')
    print('\r\tdone',len_start_time,'/',len_start_time)
    weekday=np.array(weekday).reshape((
        tt.shape[0],lt_shape[0]+1,1  
    ))
    ef=np.array(ef_list).reshape((
        tt.shape[0],lt_shape[0]+1,int(len(ef_list)/(tt.shape[0]*(lt_shape[0]+1)))
    ))
    data=dict()
    data['lt_train_tt']=train_tt[:,:lt_shape[0]*lt_shape[1]].reshape((-1,lt_shape[0],lt_shape[1],lt_shape[2]))
    data['st_train_tt']=train_tt[:,lt_shape[0]*lt_shape[1]:lt_shape[0]*lt_shape[1]+st_shape[0]]
    data['y_train_tt']=train_tt[:,int(0-op_shape[0]):]
    data['lt_test_tt']=test_tt[:,:lt_shape[0]*lt_shape[1]].reshape((-1,lt_shape[0],lt_shape[1],lt_shape[2]))
    data['st_test_tt']=test_tt[:,lt_shape[0]*lt_shape[1]:lt_shape[0]*lt_shape[1]+st_shape[0]]
    data['y_test_tt']=test_tt[:,int(0-op_shape[0]):]
    data['mms_tt']=mms_tt
    
    data['lt_train_pf']=train_pf[:,:lt_shape[0]*lt_shape[1]].reshape((-1,lt_shape[0],lt_shape[1],lt_shape[2]))
    data['st_train_pf']=train_pf[:,lt_shape[0]*lt_shape[1]:lt_shape[0]*lt_shape[1]+st_shape[0]]
    data['y_train_pf']=train_pf[:,int(0-op_shape[0]):]
    data['lt_test_pf']=test_pf[:,:lt_shape[0]*lt_shape[1]].reshape((-1,lt_shape[0],lt_shape[1],lt_shape[2]))
    data['st_test_pf']=test_pf[:,lt_shape[0]*lt_shape[1]:lt_shape[0]*lt_shape[1]+st_shape[0]]
    data['y_test_pf']=test_pf[:,int(0-op_shape[0]):]
    data['mms_pf']=mms_pf

    data['wd_train']=weekday[:spilitpoint]
    data['wd_test']=weekday[spilitpoint:]
    data['ef_train']=ef[:spilitpoint]
    data['ef_test']=ef[spilitpoint:]

    return data

def generate_data2(
    sta_num=20,
    slot_length=15,
    lt_shape=(3,9,20),
    st_shape=(8,20),
    wd_shape=(4,1),
    op_shape=(1,20),
    spilitratio=0.7
):
    print('generate data')
    tt=preprocess_data(filepath='tt_dataset.csv',savepath='tt_dataset_STDN_like.csv')
    pf=preprocess_data(filepath='pf_dataset.csv',savepath='pf_dataset_STDN_like.csv')#thr original only model ttdata
    assert len(tt)==len(pf)

    start_time=tt.pop('start_time')
    tt=tt.to_numpy().reshape((
        int(len(start_time)/(lt_shape[0]*lt_shape[1]+st_shape[0]+op_shape[0])),
        lt_shape[0]*lt_shape[1]+st_shape[0]+op_shape[0],
        sta_num
    ))
    spilitpoint=int(spilitratio*tt.shape[0])
    train_tt=tt[:spilitpoint]
    test_tt=tt[spilitpoint:]
    
    start_time=pf.pop('start_time')
    pf=pf.to_numpy().reshape((
        int(len(start_time)/(lt_shape[0]*lt_shape[1]+st_shape[0]+op_shape[0])),
        lt_shape[0]*lt_shape[1]+st_shape[0]+op_shape[0],
        sta_num
    ))
    spilitpoint=int(spilitratio*pf.shape[0])
    train_pf=pf[:spilitpoint]
    test_pf=pf[spilitpoint:]
    
    print('generate external features')
    weather_dict,wd_dict=onehot_external_features()#
    ef=pd.read_csv('weather.csv',parse_dates=['date'])
    ef.set_index(['date'],inplace=True)
    weekday=list()
    ef_list=list()
    len_start_time=len(start_time)
    for i in range(0,len_start_time,lt_shape[0]*lt_shape[1]+st_shape[0]+op_shape[0]):
        for j in range(0,lt_shape[0]*lt_shape[1]+st_shape[0],9):
            ts:pd.Timestamp=start_time[i+j]
            if ts.weekday()>=5:
                weekday.append(1)
            else:
                weekday.append(0)
            ef_list.append(ef.at[ts.date(),'AQI'])
            ef_list+=weather_dict[ef.at[ts.date(),'dw']]
            ef_list+=weather_dict[ef.at[ts.date(),'nw']]
            ef_list.append(ef.at[ts.date(),'ht'])
            ef_list.append(ef.at[ts.date(),'lt'])
            ef_list+=wd_dict[ef.at[ts.date(),'wd']]
            ef_list.append(ef.at[ts.date(),'wf'])
        print('\r\tdone',i+1,'/',len_start_time,end='')
    print('\r\tdone',len_start_time,'/',len_start_time)
    weekday=np.array(weekday).reshape((
        tt.shape[0],lt_shape[0]+1,1  
    ))
    ef=np.array(ef_list).reshape((
        tt.shape[0],lt_shape[0]+1,int(len(ef_list)/(tt.shape[0]*(lt_shape[0]+1)))
    ))
    data=dict()
    data['lt_train_tt']=train_tt[:,:lt_shape[0]*lt_shape[1]].reshape((-1,lt_shape[0],lt_shape[1],lt_shape[2]))
    data['st_train_tt']=train_tt[:,lt_shape[0]*lt_shape[1]:lt_shape[0]*lt_shape[1]+st_shape[0]]
    data['y_train_tt']=train_tt[:,int(0-op_shape[0]):]
    data['lt_test_tt']=test_tt[:,:lt_shape[0]*lt_shape[1]].reshape((-1,lt_shape[0],lt_shape[1],lt_shape[2]))
    data['st_test_tt']=test_tt[:,lt_shape[0]*lt_shape[1]:lt_shape[0]*lt_shape[1]+st_shape[0]]
    data['y_test_tt']=test_tt[:,int(0-op_shape[0]):]
    
    data['lt_train_pf']=train_pf[:,:lt_shape[0]*lt_shape[1]].reshape((-1,lt_shape[0],lt_shape[1],lt_shape[2]))
    data['st_train_pf']=train_pf[:,lt_shape[0]*lt_shape[1]:lt_shape[0]*lt_shape[1]+st_shape[0]]
    data['y_train_pf']=train_pf[:,int(0-op_shape[0]):]
    data['lt_test_pf']=test_pf[:,:lt_shape[0]*lt_shape[1]].reshape((-1,lt_shape[0],lt_shape[1],lt_shape[2]))
    data['st_test_pf']=test_pf[:,lt_shape[0]*lt_shape[1]:lt_shape[0]*lt_shape[1]+st_shape[0]]
    data['y_test_pf']=test_pf[:,int(0-op_shape[0]):]

    data['wd_train']=weekday[:spilitpoint]
    data['wd_test']=weekday[spilitpoint:]
    data['ef_train']=ef[:spilitpoint]
    data['ef_test']=ef[spilitpoint:]

    return data

def generate_data(
    sta_num=20,
    tt_slot_length=15,
    pf_slot_length=15,
    lt_shape=(3,9,20),
    st_shape=(8,20),
    wd_shape=(4,1),
    op_shape=(1,20),
    spilitratio=0.7,
    do_MinMax=True
):
    print('generate data')
    tt=preprocess_data(filepath='tt_dataset.csv',savepath='tt_dataset_STDN_like.csv',
        slot_length=tt_slot_length,lt_shape=lt_shape,st_shape=st_shape,wd_shape=wd_shape,op_shape=op_shape
    )
    pf=preprocess_data(filepath='pf_dataset.csv',savepath='pf_dataset_STDN_like.csv',
        slot_length=pf_slot_length,lt_shape=lt_shape,st_shape=st_shape,wd_shape=wd_shape,op_shape=op_shape
    )#thr original only model ttdata
    assert len(tt)==len(pf)

    start_time=tt.pop('start_time')
    tt=tt.to_numpy().reshape((
        int(len(start_time)/(lt_shape[0]*lt_shape[1]+st_shape[0]+op_shape[0])),
        lt_shape[0]*lt_shape[1]+st_shape[0]+op_shape[0],
        sta_num
    ))
    spilitpoint=int(spilitratio*tt.shape[0])
    train_tt=tt[:spilitpoint].reshape((-1,1))
    test_tt=tt[spilitpoint:].reshape((-1,1))
    mms_tt=MinMaxScaler()
    train_tt:np.ndarray=mms_tt.fit_transform(train_tt)
    train_tt=train_tt.reshape((
        spilitpoint,
        lt_shape[0]*lt_shape[1]+st_shape[0]+op_shape[0],
        sta_num
    ))
    test_tt:np.ndarray=mms_tt.transform(test_tt)
    test_tt=test_tt.reshape((
        tt.shape[0]-spilitpoint,
        lt_shape[0]*lt_shape[1]+st_shape[0]+op_shape[0],
        sta_num
    ))
    
    start_time=pf.pop('start_time')
    pf=pf.to_numpy().reshape((
        int(len(start_time)/(lt_shape[0]*lt_shape[1]+st_shape[0]+op_shape[0])),
        lt_shape[0]*lt_shape[1]+st_shape[0]+op_shape[0],
        sta_num
    ))
    spilitpoint=int(spilitratio*pf.shape[0])
    train_pf=pf[:spilitpoint].reshape((-1,1))
    test_pf=pf[spilitpoint:].reshape((-1,1))
    mms_pf=MinMaxScaler()
    train_pf:np.ndarray=mms_pf.fit_transform(train_pf)
    train_pf=train_pf.reshape((
        spilitpoint,
        lt_shape[0]*lt_shape[1]+st_shape[0]+op_shape[0],
        sta_num
    ))
    test_pf:np.ndarray=mms_pf.transform(test_pf)
    test_pf=test_pf.reshape((
        pf.shape[0]-spilitpoint,
        lt_shape[0]*lt_shape[1]+st_shape[0]+op_shape[0],
        sta_num
    ))
    
    print('generate external features')
    weather_dict,wd_dict=onehot_external_features()#
    ef=pd.read_csv('weather.csv',parse_dates=['date'])
    ef.set_index(['date'],inplace=True)
    weekday=list()
    ef_list=list()
    len_start_time=len(start_time)
    for i in range(0,len_start_time,lt_shape[0]*lt_shape[1]+st_shape[0]+op_shape[0]):
        for j in range(0,lt_shape[0]*lt_shape[1]+st_shape[0],9):
            ts:pd.Timestamp=start_time[i+j]
            if ts.weekday()>=5:
                weekday.append(1)
            else:
                weekday.append(0)
            ef_list.append(ef.at[ts.date(),'AQI'])
            ef_list+=weather_dict[ef.at[ts.date(),'dw']]
            ef_list+=weather_dict[ef.at[ts.date(),'nw']]
            ef_list.append(ef.at[ts.date(),'ht'])
            ef_list.append(ef.at[ts.date(),'lt'])
            ef_list+=wd_dict[ef.at[ts.date(),'wd']]
            ef_list.append(ef.at[ts.date(),'wf'])
        print('\r\tdone',i+1,'/',len_start_time,end='')
    print('\r\tdone',len_start_time,'/',len_start_time)
    weekday=np.array(weekday).reshape((
        tt.shape[0],lt_shape[0]+1,1  
    ))
    ef=np.array(ef_list).reshape((
        tt.shape[0],lt_shape[0]+1,int(len(ef_list)/(tt.shape[0]*(lt_shape[0]+1)))
    ))
    data=dict()
    data['lt_train_tt']=train_tt[:,:lt_shape[0]*lt_shape[1]].reshape((-1,lt_shape[0],lt_shape[1],lt_shape[2]))
    data['st_train_tt']=train_tt[:,lt_shape[0]*lt_shape[1]:lt_shape[0]*lt_shape[1]+st_shape[0]]
    data['y_train_tt']=train_tt[:,int(0-op_shape[0]):]
    data['lt_test_tt']=test_tt[:,:lt_shape[0]*lt_shape[1]].reshape((-1,lt_shape[0],lt_shape[1],lt_shape[2]))
    data['st_test_tt']=test_tt[:,lt_shape[0]*lt_shape[1]:lt_shape[0]*lt_shape[1]+st_shape[0]]
    data['y_test_tt']=test_tt[:,int(0-op_shape[0]):]
    data['mms_tt']=mms_tt
    
    data['lt_train_pf']=train_pf[:,:lt_shape[0]*lt_shape[1]].reshape((-1,lt_shape[0],lt_shape[1],lt_shape[2]))
    data['st_train_pf']=train_pf[:,lt_shape[0]*lt_shape[1]:lt_shape[0]*lt_shape[1]+st_shape[0]]
    data['y_train_pf']=train_pf[:,int(0-op_shape[0]):]
    data['lt_test_pf']=test_pf[:,:lt_shape[0]*lt_shape[1]].reshape((-1,lt_shape[0],lt_shape[1],lt_shape[2]))
    data['st_test_pf']=test_pf[:,lt_shape[0]*lt_shape[1]:lt_shape[0]*lt_shape[1]+st_shape[0]]
    data['y_test_pf']=test_pf[:,int(0-op_shape[0]):]
    data['mms_pf']=mms_pf

    data['wd_train']=weekday[:spilitpoint]
    data['wd_test']=weekday[spilitpoint:]
    data['ef_train']=ef[:spilitpoint]
    data['ef_test']=ef[spilitpoint:]

    return data

if __name__ == "__main__":
    #%matplotlib inline
    print('MultiOutput_model')
    print("\tDon't forget to\033[1;31m uncomment the %matplotlib inline \033[0m")
    print("\tDon't forget to\033[1;31m edit the epochs to 2000 when testing \033[0m")
    data=generate_data1()
    #print(data['st_train_tt'].shape)
    x_train=[data['lt_train_tt'],data['st_train_tt'],data['wd_train']]
    y_train=data['y_train_tt'].squeeze()
    x_test=[data['lt_test_tt'],data['st_test_tt'],data['wd_test']]
    y_test=data['y_test_tt'].squeeze()
    mms_tt=data['mms_tt']

    model=STDN_like2()
    earlystop=callbacks.EarlyStopping(patience=20,restore_best_weights=True)
    checkpoint=callbacks.ModelCheckpoint(model.name+'.hdf5',save_best_only=True)
    callback_list=[earlystop,checkpoint]
    clock=datetime.datetime.now()
    print('Training:',model.name,'start at:',clock)
    print_log('\n'+model.name+' start at:'+str(clock))
    model.summary(print_fn=print_log)
    history=model.fit(x_train,y_train,batch_size=16,epochs=2,callbacks=callback_list,validation_data=[x_test,y_test])
    model.save_weights(model.name+'_weight.hdf5')
    time_consumed=str(datetime.datetime.now()-clock)
    print('Training done!!!','Time consumed:',time_consumed)
    print_log('Time consumed:'+time_consumed)

    pyplot.Figure()
    pyplot.title('Metrics')
    pyplot.xlabel('epoch')
    pyplot.plot(history.history['loss'],label='loss',color='r',linestyle=':',linewidth=1.0)
    pyplot.plot(history.history['val_loss'],label='val_loss',color='r',linestyle='-',linewidth=1.0)
    pyplot.legend()
    pyplot.show()
    pyplot.savefig(model.name+'_loss.png')

    y_pred=model.predict(x_test)
    y_true=mms_tt.inverse_transform(y_test.reshape((-1,1))).flatten()
    y_pred=mms_tt.inverse_transform(y_pred.reshape((-1,1))).flatten()
    mse=mean_squared_error(y_true,y_pred)
    mae=mean_absolute_error(y_true,y_pred)
    print_log(model.name+' mse: '+str(mse))
    print_log(model.name+' mae: '+str(mae))
    print_log('\n')
    print(model.name,'mse:',mse)
    print(model.name,'mae:',mae)