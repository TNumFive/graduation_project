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

def print_log(s:str,end='\n',log_name='runtime.txt',to_stdout=False):
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

def preprocess_data2(
    filepath='tt_dataset.csv',
    savepath='tt_dataset_MyModel2.csv',
    sta_num=20,
    slot_length=15,
    lt_shape=(3,15,20),
    st_shape=(18*4,20),#72ts which means one day 
    wd_shape=(4,1),
    op_shape=(1,20)
)->pd.DataFrame:
    from datetime import datetime
    #预测第d天的t~t+op_shape[0]时隙
    #长期依赖 d-3，d-2，d-1天的 t-4～t+4时隙
    #短期依赖 d天t-8～t-1时隙,yesterday to today
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
        #make a clean to delete abnormal data
        print('delete abnormal data')
        len_raw=len(raw)
        multiplier=2
        updated=1
        counter=0
        while updated!=0:
            updated=0
            for i in range(4,24):
                i_str='%02d' %i
                for j in range(0,len_raw,72):
                    for k in range(1,71):
                        average=(raw.at[j+k-1,i_str]+raw.at[j+k+1,i_str])/2
                        if raw.at[j+k,i_str]>=average*multiplier or raw.at[j+k,i_str]<=average/multiplier:
                            raw.at[j+k,i_str]=average
                            updated=1
            counter+=1
            print('\r\t round :',counter,end='')
        print('')

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
                #short term shape will use the direct 72 slot which means a whole day
                temp:pd.Timestamp=st-pd.Timedelta(days=1)
                part=raw.loc[temp:st-pd.Timedelta(minutes=slot_length),:]
                data=data.append(part,ignore_index=True)
                temp:pd.Timestamp=st+pd.Timedelta(minutes=slot_length*op_shape[0])
                part=raw.loc[st:temp-pd.Timedelta(minutes=slot_length),:]
                data=data.append(part,ignore_index=True)
            print('\r\tpreprocessed data: ts',st,end='')
        print('')
        data.to_csv(savepath,index=False)
        print('time consumed:',datetime.now()-clock)
    return data

def generate_data(
    sta_num=20,
    slot_length=15,
    lt_shape=(3,15,20),
    st_shape=(18*4,20),
    wd_shape=(4,1),
    op_shape=(1,20),
    spilitratio=0.7
):
    print('generate data')
    tt=preprocess_data2(
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

#generate_data()

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
    print('delete abnormal data')
    len_tt=len(tt)
    multiplier=2
    updated=1
    counter=0
    while updated!=0:
        updated=0
        for i in range(4,24):
            i_str='%02d' %i
            for j in range(0,len_tt,72):
                for k in range(1,71):
                    average=(tt.at[j+k-1,i_str]+tt.at[j+k+1,i_str])/2
                    if tt.at[j+k,i_str]>=average*multiplier or tt.at[j+k,i_str]<=average/multiplier:
                        tt.at[j+k,i_str]=average
                        updated=1
        counter+=1
        print('\r\t round :',counter,end='')
    print('')
    spilitpoint=int(len_tt*0.7)
    train:pd.DataFrame=tt.iloc[:spilitpoint,:]
    test:pd.DataFrame=tt.iloc[spilitpoint:,:]

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
    print('delete abnormal data')
    len_tt=len(tt)
    multiplier=2
    updated=1
    counter=0
    while updated!=0:
        updated=0
        for i in range(4,24):
            i_str='%02d' %i
            for j in range(0,len_tt,72):
                for k in range(1,71):
                    average=(tt.at[j+k-1,i_str]+tt.at[j+k+1,i_str])/2
                    if tt.at[j+k,i_str]>=average*multiplier or tt.at[j+k,i_str]<=average/multiplier:
                        tt.at[j+k,i_str]=average
                        updated=1
        counter+=1
        print('\r\t round :',counter,end='')
    print('')
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

#mse: 970.365199942062
#mae: 21.545422153608346
historical_average_ptpd()
#mse: 971.9304525247094
#mae: 21.449440249781688
historical_average_ptpw()

def plot_line(sta_order=(4,23)):
    data=pd.read_csv('tt_dataset.csv',parse_dates=['start_time'])
    raw=pd.read_csv('tt_dataset.csv',parse_dates=['start_time'])
    #make a clean to delete abnormal data
    len_raw=len(raw)
    multiplier=2
    updated=1
    counter=0
    while updated!=0:
        updated=0
        for i in range(4,24):
            i_str='%02d' %i
            for j in range(0,len_raw,72):
                for k in range(1,71):
                    average=(raw.at[j+k-1,i_str]+raw.at[j+k+1,i_str])/2
                    if raw.at[j+k,i_str]>=average*multiplier or raw.at[j+k,i_str]<=average/multiplier:
                        raw.at[j+k,i_str]=average
                        updated=1
        counter+=1
        print('\r\t round :',counter,end='')
    print('')

    for i in range(sta_order[0],sta_order[1]+1):
        i_str='%02d' %i
        average=data.loc[:,i_str].mean()
        pyplot.figure(figsize=(84,18))
        pyplot.plot(data.loc[:,i_str],'-')
        pyplot.plot(raw.loc[:,i_str],':')
        pyplot.plot([0,len(data)],[average,average])
        pyplot.show()
        pyplot.savefig('plot_tt/sta_order_'+i_str+'.png')

#plot_line()

def multioutput():
    def convlstm2d(
        name='convlstm2d',
        input_timestep=32,
        sta_num=20,
        output_timestep=1,
        optimizer='adam',
        metrics=['mae'],
        loss='mse'
    )->Model:
        from keras.models import Sequential
        from keras.layers import BatchNormalization,LSTM,Dropout,RepeatVector,TimeDistributed,Dense,Flatten,Reshape,ConvLSTM2D

        model = Sequential(name=name)
        model.add(Reshape((input_timestep,sta_num,1,1),input_shape=(input_timestep,sta_num)))#make input same shape with purelstm model
        model.add(BatchNormalization(name = 'batch_norm_0', input_shape = (input_timestep, sta_num, 1, 1)))
        model.add(ConvLSTM2D(name ='conv_lstm_1',
                            filters = 64, kernel_size = (10, 1),                       
                            padding = 'same', 
                            return_sequences = True))
        
        model.add(Dropout(0.21, name = 'dropout_1'))
        model.add(BatchNormalization(name = 'batch_norm_1'))

        model.add(ConvLSTM2D(name ='conv_lstm_2',
                            filters = 64, kernel_size = (5, 1), 
                            padding='same',
                            return_sequences = False))
        
        model.add(Dropout(0.20, name = 'dropout_2'))
        model.add(BatchNormalization(name = 'batch_norm_2'))
        
        model.add(Flatten())
        model.add(RepeatVector(output_timestep))
        model.add(Reshape((output_timestep, sta_num, 1, 64)))
        
        model.add(ConvLSTM2D(name ='conv_lstm_3',
                            filters = 64, kernel_size = (10, 1), 
                            padding='same',
                            return_sequences = True))
        
        model.add(Dropout(0.20, name = 'dropout_3'))
        model.add(BatchNormalization(name = 'batch_norm_3'))
        
        model.add(ConvLSTM2D(name ='conv_lstm_4',
                            filters = 64, kernel_size = (5, 1), 
                            padding='same',
                            return_sequences = True))
        
        model.add(TimeDistributed(Dense(units=1, name = 'dense_1', activation = 'relu')))
        model.add(Flatten())#prevent topological error
        
        model.compile(optimizer=optimizer,metrics=metrics,loss=loss)
        
        return model
    model=convlstm2d()
    data=generate_data()
    x_train=data['st_train_tt'][:,40:]
    y_train=data['y_train_tt'].squeeze()
    x_test=data['st_test_tt'][:,40:]
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

#multioutput()

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

    y1=layers.TimeDistributed(layers.Conv1D(filters=64,kernel_size=6,padding='same'))(lt)
    y1=layers.ReLU()(y1)
    y1=layers.TimeDistributed(layers.Conv1D(filters=64,kernel_size=6,padding='same'))(y1)
    y1=layers.ReLU()(y1)
    y1=layers.Activation('sigmoid')(y1)
    y1=layers.TimeDistributed(layers.LSTM(64,return_sequences=True))(y1)
    y1=layers.TimeDistributed(attention())(y1)
    y1=attention()(y1)
    

    y2=layers.Conv1D(filters=64,kernel_size=6,padding='same')(st)
    y2=layers.ReLU()(y2)
    y2=layers.Conv1D(filters=64,kernel_size=6,padding='same')(y2)
    y2=layers.ReLU()(y2)
    y2=layers.Activation('sigmoid')(y2)
    y2=layers.LSTM(64,return_sequences=True)(y2)
    y2=attention()(y2)
    
    y=layers.concatenate([y1,y2])
    y=layers.Dense(sta_num)(y)

    model=Model([lt,st,lt_wd,st_wd],[y],name='MyModel')
    model.compile(optimizer=optimizer,loss=loss,metrics=metrics)
    return model

'''
very close to convlstm on st on mse and slightly better than that on mae
def MyModel_Conv2D(
    name='MyModel_Conv2D',
    sta_num=20,
    lt_shape=(3,15,20),
    st_shape=(72,20),
    wd_shape=(4,1),
    ef_shape=(4,24),
    op_shape=(1,20),
    optimizer='adam',
    metrics=['mae'],
    loss='mse'
)->Model:
  print('model name:',name)
  lt=layers.Input(shape=lt_shape)
  st=layers.Input(shape=st_shape)
  lt_wd=layers.Input(shape=(lt_shape[0],wd_shape[1]))
  st_wd=layers.Input(shape=(wd_shape[1],))
  lt_ef=layers.Input(shape=(lt_shape[0],ef_shape[1]))
  st_ef=layers.Input(shape=(ef_shape[1],))

  y1=layers.Reshape((lt_shape[0],lt_shape[1],lt_shape[2],1))(lt)
  y1=layers.ConvLSTM2D(filters=64,kernel_size=(6,6),padding='same',return_sequences=True)(y1)
  y1=layers.Dropout(0.5)(y1)
  y1=layers.ConvLSTM2D(filters=64,kernel_size=(6,6),padding='same',return_sequences=True)(y1)
  y1=layers.Dropout(0.25)(y1)
  y1=layers.Dense(1)(y1)
  y1=layers.Reshape(lt_shape)(y1)
  y1=layers.TimeDistributed(attention())(y1)
  y1=layers.Flatten()(y1)

  y2=layers.Reshape((st_shape[0],st_shape[1],1))(st)
  y2=layers.Conv2D(filters=64,kernel_size=(6,6),padding='same')(y2)
  y2=layers.ReLU()(y2)
  y2=layers.Conv2D(filters=64,kernel_size=(6,6),padding='same')(y2)
  y2=layers.ReLU()(y2)
  y2=layers.Dense(1)(y2)
  y2=layers.Reshape(st_shape)(y2)
  y2=attention()(y2)

  ltef=layers.concatenate([lt_wd,lt_ef])
  stef=layers.concatenate([st_wd,st_ef])
  stef=layers.Reshape((1,wd_shape[1]+ef_shape[1]))(stef)
  ef=layers.concatenate([ltef,stef],axis=1)
  ef=layers.LSTM(64,return_sequences=True)(ef)
  ef=layers.Dropout(0.2)(ef)
  ef=layers.LSTM(64,return_sequences=True)(ef)
  ef=layers.Dropout(0.2)(ef)
  ef=layers.ReLU()(ef)
  ef=attention()(ef)
  
  y=layers.concatenate([y1,y2,ef])
  y=layers.Dense(sta_num)(y)

  model=Model([lt,st,lt_wd,st_wd,lt_ef,st_ef],[y],name=name)
  model.compile(optimizer=optimizer,loss=loss,metrics=metrics)
  
  return model

'''

def MyModel_Conv2D(
    name='MyModel_Conv2D',
    sta_num=20,
    lt_shape=(3,15,20),
    st_shape=(72,20),
    wd_shape=(4,1),
    ef_shape=(4,24),
    op_shape=(1,20),
    optimizer='adam',
    metrics=['mae'],
    loss='mse'
)->Model:
    print('model name:',name)
    lt=layers.Input(shape=lt_shape)
    st=layers.Input(shape=st_shape)
    lt_wd=layers.Input(shape=(lt_shape[0],wd_shape[1]))
    st_wd=layers.Input(shape=(wd_shape[1],))
    lt_ef=layers.Input(shape=(lt_shape[0],ef_shape[1]))
    st_ef=layers.Input(shape=(ef_shape[1],))

    y1=layers.Reshape((lt_shape[0],lt_shape[1],lt_shape[2],1))(lt)
    y1=layers.ConvLSTM2D(filters=64,kernel_size=(6,6),padding='same',return_sequences=True)(y1)
    y1=layers.Dropout(0.5)(y1)
    y1=layers.ConvLSTM2D(filters=64,kernel_size=(6,6),padding='same',return_sequences=True)(y1)
    y1=layers.Dropout(0.25)(y1)
    y1=layers.Dense(1)(y1)
    y1=layers.Reshape(lt_shape)(y1)
    y1=layers.TimeDistributed(attention())(y1)
    y1=layers.Flatten()(y1)

    y2=layers.Reshape((st_shape[0],st_shape[1],1))(st)
    y2=layers.Conv2D(filters=64,kernel_size=(6,6),padding='same')(y2)
    y2=layers.ReLU()(y2)
    y2=layers.Conv2D(filters=64,kernel_size=(6,6),padding='same')(y2)
    y2=layers.ReLU()(y2)
    y2=layers.Dense(1)(y2)
    y2=layers.Reshape(st_shape)(y2)
    y2=attention()(y2)

    ltef=layers.concatenate([lt_wd,lt_ef])
    stef=layers.concatenate([st_wd,st_ef])
    stef=layers.Reshape((1,wd_shape[1]+ef_shape[1]))(stef)
    ef=layers.concatenate([ltef,stef],axis=1)
    ef=layers.LSTM(64,return_sequences=True)(ef)
    ef=layers.Dropout(0.2)(ef)
    ef=layers.LSTM(64,return_sequences=True)(ef)
    ef=layers.Dropout(0.2)(ef)
    ef=layers.ReLU()(ef)
    ef=attention()(ef)
    
    y=layers.concatenate([y1,y2,ef])
    y=layers.Dense(sta_num)(y)

    model=Model([lt,st,lt_wd,st_wd,lt_ef,st_ef],[y],name='MyModel')
    model.compile(optimizer=optimizer,loss=loss,metrics=metrics)
    return model

model=MyModel_Conv2D()
plot_model(model,model.name+'.png',show_shapes=True)

if __name__ == "__man__":
    #%matplotlib inline
    print("\tDon't forget to\033[1;31m uncomment the %matplotlib inline \033[0m")
    print("\tDon't forget to\033[1;31m edit the epochs to 2000 when testing \033[0m")
    
    data=generate_data()
    print('data shpae:')
    print(data['lt_train_tt'].shape,data['st_train_tt'].shape,data['y_train_tt'].shape,data['lt_wd_train'].shape,data['st_wd_train'].shape,data['lt_ef_train'].shape,data['st_ef_train'].shape)
    print(data['lt_test_tt'].shape,data['st_test_tt'].shape,data['y_test_tt'].shape,data['lt_wd_test'].shape,data['st_wd_test'].shape,data['lt_ef_test'].shape,data['st_ef_test'].shape)
    x_train=[data['lt_train_tt'],data['st_train_tt'],data['lt_wd_train'],data['st_wd_train'],data['lt_ef_train'],data['st_ef_train']]
    y_train=data['y_train_tt'].squeeze()
    x_test=[data['lt_test_tt'],data['st_test_tt'],data['lt_wd_test'],data['st_wd_test'],data['lt_ef_test'],data['st_ef_test']]
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
