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
from sklearn.preprocessing import MinMaxScaler,OneHotEncoder
from sklearn.metrics import mean_absolute_error,mean_squared_error

np.random.seed(seed=5)

def print_log(s:str,end='\n',log_name='runtime.log'):
    runtime=open(log_name,'a+')
    runtime.write(s+end)
    runtime.close()

def purelstm(
    name='purelstm',
    input_timestep=32,
    sta_num=20,
    output_timestep=1,
    optimizer='adam',
    metrics=['mae'],
    loss='mse'
)->Model:
    from keras.models import Sequential
    from keras.layers import BatchNormalization,LSTM,Dropout,RepeatVector,TimeDistributed,Dense,Flatten

    model = Sequential(name=name)
    model.add(BatchNormalization(name = 'batch_norm_0', input_shape = (input_timestep, sta_num)))
    model.add(LSTM(name ='lstm_1',
                   units = 64,
                   return_sequences = True))
    
    model.add(Dropout(0.2, name = 'dropout_1'))
    model.add(BatchNormalization(name = 'batch_norm_1'))

    model.add(LSTM(name ='lstm_2',
                   units = 64,
                   return_sequences = False))
    
    model.add(Dropout(0.1, name = 'dropout_2'))
    model.add(BatchNormalization(name = 'batch_norm_2'))
    
    model.add(RepeatVector(output_timestep))
    
    model.add(LSTM(name ='lstm_3',
                   units = 64,
                   return_sequences = True))
    
    model.add(Dropout(0.1, name = 'dropout_3'))
    model.add(BatchNormalization(name = 'batch_norm_3'))
    
    model.add(LSTM(name ='lstm_4',
                   units = sta_num,
                   return_sequences = True))
    
    model.add(TimeDistributed(Dense(units=sta_num, name = 'dense_1', activation = 'linear')))
    model.add(Flatten())#prevent topological error 

    model.compile(optimizer=optimizer,metrics=metrics,loss=loss)

    return model

def purelstm_ori(
    name='purelstm_ori',
    input_timestep=32,
    sta_num=20,
    output_timestep=1,
    optimizer='adam',
    metrics=['mae'],
    loss='mse'
)->Model:
    from keras.models import Sequential
    from keras.layers import BatchNormalization,LSTM,Dropout,RepeatVector,TimeDistributed,Dense,Flatten

    model = Sequential(name=name)
    model.add(BatchNormalization(name = 'batch_norm_0', input_shape = (input_timestep, sta_num)))
    model.add(LSTM(name ='lstm_1',
                   units = 64,
                   return_sequences = True))
    
    model.add(Dropout(0.2, name = 'dropout_1'))
    model.add(BatchNormalization(name = 'batch_norm_1'))

    model.add(LSTM(name ='lstm_2',
                   units = 64,
                   return_sequences = False))
    
    model.add(Dropout(0.1, name = 'dropout_2'))
    model.add(BatchNormalization(name = 'batch_norm_2'))
    
    model.add(RepeatVector(output_timestep))

    model.add(TimeDistributed(Dense(units=sta_num, name = 'dense_1', activation = 'linear')))
    model.add(Flatten())#prevent topological error 

    model.compile(optimizer=optimizer,metrics=metrics,loss=loss)

    return model

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
    savepath='tt_dataset_MultiOutput.csv',
    #is_tt=1,
    input_timestep=32,
    #sta_num=20,
    output_timestep=1
)->pd.DataFrame:
    print('preprocess data',filepath)
    file_list=os.listdir('./')
    if savepath in file_list:
        data=pd.read_csv(savepath,parse_dates=['start_time'])
    else:
        data=pd.DataFrame()
        raw=pd.read_csv(filepath,parse_dates=['start_time'])
        raw.set_index(['start_time'],inplace=True,drop=False)
        len_raw=len(raw)
        for i in range(input_timestep+1,len_raw):
            start:pd.Timestamp=raw.iloc[i-input_timestep-1].name
            now:pd.Timestamp=raw.iloc[i].name
            if i+output_timestep-1<len_raw:
                end:pd.Timestamp=raw.iloc[i+output_timestep-1].name
            else:
                end=now+pd.Timedelta(days=1)
            if (now.day-start.day<=1) and (now.day==end.day):#cause the timerange was 2018-12-1~2018-12-30 2019-01-11~2019-03-20
                data=data.append(raw.iloc[i-input_timestep-1:i+output_timestep-1],ignore_index=True)
            print('\r\tpreprocessing data:',i+1,'/',len_raw,end='')
        print('')
        data.to_csv(savepath,index=False)
    return data

def generate_data(
    input_timestep=32,
    sta_num=20,
    output_timestep=1,
    spilitratio=0.7
):
    print('generate data')
    #data format 32 step in with 1 step out
    tt=preprocess_data(filepath='tt_dataset.csv',savepath='tt_dataset_MultiOutput.csv')
    pf=preprocess_data(filepath='pf_dataset.csv',savepath='pf_dataset_MultiOutput.csv')#thr original only model ttdata
    #weather_dict,wd_dict=onehot_external_features()#the original model only use short term data
    start_time=tt.pop('start_time')
    tt=tt.to_numpy().reshape((
        int(len(start_time)/(input_timestep+output_timestep)),input_timestep+output_timestep,sta_num
    ))
    spilitpoint=int(spilitratio*tt.shape[0])
    train_tt=tt[:spilitpoint].reshape((-1,1))
    test_tt=tt[spilitpoint:].reshape((-1,1))
    mms_tt=MinMaxScaler()
    train_tt:np.ndarray=mms_tt.fit_transform(train_tt)
    train_tt=train_tt.reshape((
        spilitpoint,input_timestep+output_timestep,sta_num
    ))
    test_tt:np.ndarray=mms_tt.transform(test_tt)
    test_tt=test_tt.reshape((
        tt.shape[0]-spilitpoint,input_timestep+output_timestep,sta_num
    ))
    
    start_time=pf.pop('start_time')
    pf=pf.to_numpy().reshape((
        int(len(start_time)/(input_timestep+output_timestep)),input_timestep+output_timestep,sta_num
    ))
    spilitpoint=int(spilitratio*pf.shape[0])
    train_pf=pf[:spilitpoint].reshape((-1,1))
    test_pf=pf[spilitpoint:].reshape((-1,1))
    mms_pf=MinMaxScaler()
    train_pf:np.ndarray=mms_pf.fit_transform(train_pf)
    train_pf=train_pf.reshape((
        spilitpoint,input_timestep+output_timestep,sta_num
    ))
    test_pf:np.ndarray=mms_pf.transform(test_pf)
    test_pf=test_pf.reshape((
        pf.shape[0]-spilitpoint,input_timestep+output_timestep,sta_num
    ))
    data=dict()
    data['st_train_tt']=train_tt[:,:input_timestep,:]
    data['y_train_tt']=train_tt[:,input_timestep:,:]
    data['st_test_tt']=test_tt[:,:input_timestep,:]
    data['y_test_tt']=test_tt[:,input_timestep:,:]
    data['mms_tt']=mms_tt
    data['st_train_pf']=train_pf[:,:input_timestep,:]
    data['y_train_pf']=train_pf[:,input_timestep:,:]
    data['st_test_pf']=test_pf[:,:input_timestep,:]
    data['y_test_pf']=test_pf[:,input_timestep:,:]
    data['mms_pf']=mms_pf

    return data
    
if __name__ == "__main__":
    #%matplotlib inline
    print('MultiOutput_model')
    print("\tDon't forget to\033[1;31m uncomment the %matplotlib inline \033[0m")
    print("\tDon't forget to\033[1;31m edit the epochs to 2000 when testing \033[0m")
    data=generate_data()
    print(data['st_train_tt'].shape)
    x_train=data['st_train_tt']
    y_train=data['y_train_tt'].squeeze()
    x_test=data['st_test_tt']
    y_test=data['y_test_tt'].squeeze()
    mms_tt=data['mms_tt']

    model=purelstm_ori()
    earlystop=callbacks.EarlyStopping(patience=20,restore_best_weights=True)
    checkpoint=callbacks.ModelCheckpoint(model.name+'.hdf5',save_best_only=True)
    callback_list=[earlystop,checkpoint]
    clock=datetime.datetime.now()
    print('Training:',model.name,'start at:',clock)
    print_log('\n'+model.name+' start at:'+str(clock))
    model.summary(print_fn=print_log)
    history=model.fit(x_train,y_train,batch_size=16,epochs=2000,callbacks=callback_list,validation_data=[x_test,y_test])
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