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

def print_log(s:str,end='\n',log_name='runtime.log',to_stdout=False):
    runtime=open(log_name,'a+')
    runtime.write(s+end)
    runtime.close()
    if to_stdout:
        print(s,end=end)

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

class attention(layers.Layer):#simplest attention layer
    def __init__(self, **kwargs):
        super(attention, self).__init__(**kwargs)

    def build(self,input_shape):#input_shape=(,9,64)
        assert len(input_shape)==3
        self.W=self.add_weight(name='attr_weight',shape=(input_shape[1],input_shape[2]),initializer='uniform',trainable=True)
        self.b=self.add_weight(name='attr_bias',shape=(input_shape[2],),initializer='uniform',trainable=True)
        super(attention,self).build(input_shape)

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

def preprocess_data(
    filepath='tt_dataset.csv',
    savepath='tt_dataset_MyModel.csv',
    sta_num=20,
    slot_length=15,
    lt_shape=(3,15,20),
    st_shape=(7,20),
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
                part.fillna(value=0,inplace=True)
                data=data.append(part,ignore_index=True)
            print('\r\tpreprocessing data: ts',st,end='')
        print('')
        data.to_csv(savepath,index=False)
        print('time consumed:',datetime.now()-clock)
    return data

def generate_data(
    sta_num=20,
    slot_length=15,
    lt_shape=(3,15,20),
    st_shape=(7,20),
    wd_shape=(4,1),
    op_shape=(1,20),
    spilitratio=0.7
):
    print('generate data')
    tt=preprocess_data(
        slot_length=slot_length,
        lt_shape=lt_shape,
        st_shape=st_shape,
        wd_shape=wd_shape,
        op_shape=op_shape
    )

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
    
    print('generate external features')
    weather_dict,wd_dict=onehot_external_features()#
    ef=pd.read_csv('weather.csv',parse_dates=['date'])
    ef.set_index(['date'],inplace=True)
    weekday=list()
    ef_list=list()
    len_start_time=len(start_time)
    for i in range(0,len_start_time,lt_shape[0]*lt_shape[1]+st_shape[0]+op_shape[0]):
        for j in range(0,lt_shape[0]*lt_shape[1]+1,lt_shape[1]):
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
        -1,lt_shape[0]+1,1  
    ))
    ef=np.array(ef_list).reshape((
        -1,lt_shape[0]+1,int(len(ef_list)/(tt.shape[0]*(lt_shape[0]+1)))
    ))
    data=dict()
    data['lt_train_tt']=train_tt[:,:lt_shape[0]*lt_shape[1]].reshape((-1,lt_shape[0],lt_shape[1],lt_shape[2]))
    data['st_train_tt']=train_tt[:,lt_shape[0]*lt_shape[1]:lt_shape[0]*lt_shape[1]+st_shape[0]]
    data['y_train_tt']=train_tt[:,int(0-op_shape[0]):]
    data['lt_test_tt']=test_tt[:,:lt_shape[0]*lt_shape[1]].reshape((-1,lt_shape[0],lt_shape[1],lt_shape[2]))
    data['st_test_tt']=test_tt[:,lt_shape[0]*lt_shape[1]:lt_shape[0]*lt_shape[1]+st_shape[0]]
    data['y_test_tt']=test_tt[:,int(0-op_shape[0]):]
    data['mms_tt']=mms_tt

    data['lt_wd_train']=weekday[:spilitpoint,:-1]
    data['st_wd_train']=weekday[:spilitpoint,-1]
    data['lt_wd_test']=weekday[spilitpoint:,:-1]
    data['st_wd_test']=weekday[spilitpoint:,-1]
    data['lt_ef_train']=ef[:spilitpoint,:-1]
    data['st_ef_train']=ef[:spilitpoint,-1]
    data['lt_ef_test']=ef[spilitpoint:,:-1]
    data['st_ef_test']=ef[spilitpoint:,-1]

    return data

def historical_average_ptpd(
    sta_num=20,
    slot_length=15,
    lt_shape=(3,15,20),
    st_shape=(7,20),
    wd_shape=(4,1),
    op_shape=(1,20)
):
    print('historical averge by per ts per day ')
    tt=pd.read_csv('tt_dataset.csv',parse_dates=['start_time'])
    len_tt=len(tt)
    spilitpoint=int(len_tt*0.7)
    train:pd.DataFrame=tt.iloc[:spilitpoint,:]
    test:pd.DataFrame=tt.iloc[spilitpoint:,:]

    file_list=os.listdir('./')
    if 'haptpd.csv' not in file_list:
        ans=pd.DataFrame()
        train.set_index(['start_time'],inplace=True)
        start=pd.to_datetime('5:15:00')
        end=pd.to_datetime('23:00:00')
        time=start
        print('generate ha data')
        while time<=end:
            print('\r\tworking on :',time,end='')
            time_str=str(time)[-8:]
            time+=pd.Timedelta(minutes=slot_length)
            slot:pd.Dataframe=train.at_time(time_str)
            slot:pd.Series=slot.mean()
            slot['start_time']=time_str
            ans=ans.append(slot,ignore_index=True)
        print('\ndone')
        ans.to_csv('haptpd.csv',index=False)
    else:
        ans=pd.read_csv('haptpd.csv')    
        ans.set_index(['start_time'],inplace=True)
    test_index=test.pop('start_time')
    pred=pd.DataFrame()
    for index in test_index:
        print('\r\t append',index,end='')
        pred=pred.append(ans.loc[str(index)[-8:],:],ignore_index=True)
    print('')
    print('done')
    true=test.to_numpy().flatten()
    pred=pred.to_numpy().flatten()
    mse=mean_squared_error(true,pred)
    mae=mean_absolute_error(true,pred)
    print('mse:',mse)
    print('mae:',mae)

def historical_average_ptpw(
    sta_num=20,
    slot_length=15,
    lt_shape=(3,15,20),
    st_shape=(7,20),
    wd_shape=(4,1),
    op_shape=(1,20)
):
    print('historical averge by per ts per weekday ')
    tt=pd.read_csv('tt_dataset.csv',parse_dates=['start_time'])
    len_tt=len(tt)
    spilitpoint=int(len_tt*0.7)
    train:pd.DataFrame=pd.DataFrame(tt.loc[:spilitpoint,:])
    test:pd.DataFrame=pd.DataFrame(tt.loc[spilitpoint:,:])
    
    key_dict=dict()
    for i in range(0,len(train)):
        time:pd.Timestamp=train.iloc[i]['start_time']
        key=time.hour*100+time.minute
        if time.weekday()<=4:
            key=key+10000
        train.at[i,'key']=key
        if key not in key_dict.keys():
            key_dict[key]=1
        else:
            key_dict[key]+=1
        print('\r\t generate key:',i+1,'/',spilitpoint,end='')
    print('')
    train.drop(columns=['start_time'])
    ans=pd.DataFrame()
    for i in key_dict.keys():
        slot=train.query('key==@i')
        slot=slot.mean()
        slot['key']=i
        ans=ans.append(slot,ignore_index=True)
        print('\r\t generate ha:',i,end='')
    print('')
    ans.set_index(['key'],inplace=True)

    test_index=test.pop('start_time')
    pred=pd.DataFrame()
    for index in test_index:
        print('\r\t append',index,end='')
        key=index.hour*100+index.minute
        if index.weekday()<=4:
            key+=10000
        pred=pred.append(ans.loc[key,:],ignore_index=True)
    print('')
    print('done')
    true=test.to_numpy().flatten()
    pred=pred.to_numpy().flatten()
    mse=mean_squared_error(true,pred)
    mae=mean_absolute_error(true,pred)
    print('mse:',mse)
    print('mae:',mae)

historical_average_ptpd()
historical_average_ptpw()
    

def MyModel(
    name='MyModel',
    sta_num=20,
    lt_shape=(3,15,20),
    st_shape=(7,20),
    wd_shape=(4,1),
    op_shape=(1,20),
    optimizer='adam',
    metrics=['mae'],
    loss='mse'
)->Model:

    print('build model:',name)
    lt=layers.Input(shape=lt_shape)
    st=layers.Input(shape=st_shape)
    lt_wd=layers.Input(shape=(lt_shape[0],wd_shape[1]))
    st_wd=layers.Input(shape=(wd_shape[1],))

    y11=layers.TimeDistributed(layers.Conv1D(filters=64,kernel_size=6,padding='same'))(lt)
    y11=layers.ReLU()(y11)
    y12=layers.TimeDistributed(layers.LSTM(64,return_sequences=True))(y11)
    y12=layers.TimeDistributed(attention())(y12)
    y12=layers.ReLU()(y12)
    y12=layers.concatenate([y12,lt_wd])
    y13=layers.LSTM(64,return_sequences=True)(y12)
    y13=attention()(y13)

    y21=layers.Conv1D(filters=64,kernel_size=6,padding='same')(st)
    y21=layers.ReLU()(y21)
    wd=layers.RepeatVector(st_shape[0])(st_wd)
    y22=layers.concatenate([y21,wd])
    y22=layers.LSTM(64,return_sequences=True)(y22)
    y22=attention()(y22)
    
    y=layers.concatenate([y13,y22])
    y=layers.ReLU()(y)
    y=layers.Dense(20)(y)

    model=Model([lt,st,lt_wd,st_wd],[y],name='MyModel')
    model.compile(optimizer=optimizer,loss=loss,metrics=metrics)
    return model

if __name__ == "__main__":
    #%matplotlib inline
    '''
    print("\tDon't forget to\033[1;31m uncomment the %matplotlib inline \033[0m")
    print("\tDon't forget to\033[1;31m edit the epochs to 2000 when testing \033[0m")

    model=MyModel()
    plot_model(model,'MyModel.png',show_shapes=True)

    data=generate_data()
    print('data shpae:')
    print(data['lt_train_tt'].shape,data['st_train_tt'].shape,data['y_train_tt'].shape,data['lt_wd_train'].shape,data['st_wd_train'].shape,data['lt_ef_train'].shape,data['st_ef_train'].shape)
    print(data['lt_test_tt'].shape,data['st_test_tt'].shape,data['y_test_tt'].shape,data['lt_wd_test'].shape,data['st_wd_test'].shape,data['lt_ef_test'].shape,data['st_ef_test'].shape)
    x_train=[data['lt_train_tt'],data['st_train_tt'],data['lt_wd_train'],data['st_wd_train']]
    y_train=data['y_train_tt'].squeeze()
    x_test=[data['lt_test_tt'],data['st_test_tt'],data['lt_wd_test'],data['st_wd_test']]
    y_test=data['y_test_tt'].squeeze()
    mms_tt=data['mms_tt']

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
    print_log(model.name+' mse: '+str(mse),to_stdout=True)
    print_log(model.name+' mae: '+str(mae),to_stdout=True)
    print_log('\n')
    '''