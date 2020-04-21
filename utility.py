import numpy as np
import pandas as pd 
import os
import datetime

np.random.seed(seed=5)

from matplotlib import pyplot
from numpy import ndarray as nda


from keras import Model
from keras.models import save_model,load_model
from keras.callbacks import Callback,EarlyStopping,ReduceLROnPlateau
from keras.losses import mean_absolute_percentage_error,mean_absolute_error,mean_squared_error
from keras import backend as K
from keras.layers import Layer
from sklearn.preprocessing import OneHotEncoder

def print_log(s:str,end='\n',log_name='runtime.log'):
        runtime=open(log_name,'a+')
        runtime.write(s+end)
        runtime.close()

class CustomModelCheckpoint(Callback):
    def __init__(self, dirpath, filepath, monitor='val_loss', 
                 mode='auto', period=1,verbose=1):
        super(CustomModelCheckpoint, self).__init__()
        self.monitor = monitor
        self.dirpath = dirpath
        self.filepath=filepath
        self.period = period
        self.epochs_since_last_save = 0
        self.best_logs={}
        self.best_epoch=0
        self.best_weight=None
        self.verbose=verbose

        if mode not in ['auto', 'min', 'max']:
            mode = 'auto'
        if mode == 'min':
            self.monitor_op = np.less
            self.best = np.Inf
        elif mode == 'max':
            self.monitor_op = np.greater
            self.best = -np.Inf
        else:
            if 'acc' in self.monitor or self.monitor.startswith('fmeasure'):
                self.monitor_op = np.greater
                self.best = -np.Inf
            else:
                self.monitor_op = np.less
                self.best = np.Inf

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.epochs_since_last_save+=1
        current=logs[self.monitor]
        if self.monitor_op(current,self.best):#better one appeared
            self.best=current
            self.update=1
            save_model(self.model,self.dirpath+'model.hdf5')
            self.best_epoch=epoch
            self.best_weight=self.model.get_weights()
            for k in logs.keys():
                self.best_logs[k]=logs[k]
        if self.epochs_since_last_save>=self.period:#time to save 
            self.epochs_since_last_save=0
            if self.update==1:
                self.update=0
                path=self.filepath.format(epoch=self.best_epoch+1,**(self.best_logs))
                os.rename(self.dirpath+'model.hdf5',self.dirpath+path)
                if self.verbose==1:
                    print('save to:',path)
            if epoch!=self.best_epoch:
                self.model.set_weights(self.best_weight)
    
    def on_train_end(self,logs=None):
        if self.update==1:#updated but not saved
            path=self.filepath.format(epoch=self.best_epoch+1,**(self.best_logs))
            os.rename(self.dirpath+'model.hdf5',self.dirpath+path)
            if self.verbose==1:
                print('save to:',path)
        print_log('')#start a new line
        print_log('Timestamp:'+str(datetime.datetime.now()))
        self.model.summary(print_fn=print_log)
        print_log(self.filepath.format(epoch=self.best_epoch+1,**(self.best_logs)))

#loss function
def rmse(y_true,y_pred):
    return K.sqrt(mean_squared_error(y_true,y_pred))
def mape(y_true,y_pred):
    return mean_absolute_percentage_error(y_true,y_pred)/100.0
def mae(y_true,y_pred):
    return mean_absolute_error(y_true,y_pred)*60.0/100.0

def onehot_external_features()->(dict,dict):
    ef=pd.read_csv('./data/weather.csv',parse_dates=['date'])
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
    weather_array=np.array(weather_list).reshape(-1,1)
    wd_array=np.array(wd_list).reshape(-1,1)
    weather_array=enc.fit_transform(weather_array).toarray()
    wd_array=enc.fit_transform(wd_array).toarray()
    weather_dict=dict()
    for i in range(0,len(weather_array)):
        weather_dict[weather_list[i]]=weather_array[i]
    wd_dict=dict()
    for i in range(0,len(wd_array)):
        wd_dict[wd_list[i]]=wd_array[i]
    return weather_dict,wd_dict

def custom_scale2(fore_step=4,output_step=1,sta_num=22,feature=1,ratio=0.7)->(nda,nda,nda,nda,nda,nda,nda,nda,nda,nda):
    st=pd.read_csv('./data/temp_dataset_st.csv',parse_dates=['start_time'])
    lt=pd.read_csv('./data/temp_dataset_lt.csv',parse_dates=['start_time'])
    se=pd.read_csv('./data/temp_dataset_se.csv',parse_dates=['date'])
    le=pd.read_csv('./data/temp_dataset_le.csv',parse_dates=['date'])
    op=pd.read_csv('./data/temp_dataset_op.csv',parse_dates=['start_time'])
    st.set_index(['start_time'],inplace=True)
    lt.set_index(['start_time'],inplace=True)
    #se.set_index(['date'],inplace=True)#this might be a problem as one date might appear few times
    se.drop(columns=['date','dw','nw','wd'],inplace=True)#in case of new external features
    #le.set_index(['date'],inplace=True)
    le.drop(columns=['date','dw','nw','wd'],inplace=True)
    op.set_index(['start_time'],inplace=True)
    
    splitpoint=int(len(op)/output_step*ratio)
    #if splitpoint>=20*4*7:
    #    splitpoint=20*4*7#just predict the last week
    data=st.to_numpy().reshape((int(len(st)/fore_step),fore_step,sta_num,feature))/60.0
    st_train=data[:splitpoint]
    st_test=data[splitpoint:]
    print('short term',data.shape)
    #print(data[0])

    data=lt.to_numpy().reshape((int(len(lt)/fore_step),fore_step,sta_num,feature))/60.0
    lt_train=data[:splitpoint]
    lt_test=data[splitpoint:]
    print('long term',data.shape)
    #print(data[0])

    se.loc[:,'AQI']=np.ceil(se.loc[:,'AQI']/50.0)
    data=se.to_numpy()
    se_train=data[:splitpoint]
    se_test=data[splitpoint:]
    print('short external feature',data.shape)
    #print(data[0])

    le.loc[:,'AQI']=np.ceil(le.loc[:,'AQI']/50.0)
    data=le.to_numpy().reshape((int(len(le)/fore_step),fore_step,len(le.columns)))
    le_train=data[:splitpoint]
    le_test=data[splitpoint:]
    print('long external feature',data.shape)
    #print(data[0])    

    data=op.to_numpy()/60.0
    op_train=data[:splitpoint]
    op_test=data[splitpoint:]
    print('output',data.shape)
    #print(data[0])

    return st_train,st_test,lt_train,lt_test,se_train,se_test,le_train,le_test,op_train,op_test

def custom_scale3(fore_step=4,output_step=1,sta_num=22,feature=1,ratio=0.7)->(nda,nda,nda,nda,nda,nda,nda,nda,nda,nda):
    st=pd.read_csv('./data/temp_dataset_st.csv',parse_dates=['start_time'])
    lt=pd.read_csv('./data/temp_dataset_lt.csv',parse_dates=['start_time'])
    se=pd.read_csv('./data/temp_dataset_se.csv',parse_dates=['date'])
    le=pd.read_csv('./data/temp_dataset_le.csv',parse_dates=['date'])
    op=pd.read_csv('./data/temp_dataset_op.csv',parse_dates=['start_time'])
    st.set_index(['start_time'],inplace=True)
    lt.set_index(['start_time'],inplace=True)
    #se.set_index(['date'],inplace=True)
    #le.set_index(['date'],inplace=True)
    se.drop(columns=['date'],inplace=True)
    le.drop(columns=['date'],inplace=True)
    op.set_index(['start_time'],inplace=True)

    #one-hot encode str-like external feature
    weather_dict,wd_dict=onehot_external_features()
    #split weather&wind direction array
    for i in range(0,len(se)):
        se.at[i,'dw']=weather_dict[se.at[i,'dw']]
        se.at[i,'nw']=weather_dict[se.at[i,'nw']]
        se.at[i,'wd']=wd_dict[se.at[i,'wd']]
    for i in range(0,len(se.at[i,'dw'])):
        i_str='%02d' %i
        se.loc[:,'dw'+i_str]=se.loc[:,'dw'].map(lambda x: x[i])
        se.loc[:,'nw'+i_str]=se.loc[:,'nw'].map(lambda x: x[i])
    for i in range(0,len(se.at[i,'wd'])):
        i_str='%02d' %i
        se.loc[:,'wd'+i_str]=se.loc[:,'wd'].map(lambda x: x[i])
    se.drop(columns=['dw','nw','wd'],inplace=True)
    for i in range(0,len(le)):
        le.at[i,'dw']=weather_dict[le.at[i,'dw']]
        le.at[i,'nw']=weather_dict[le.at[i,'nw']]
        le.at[i,'wd']=wd_dict[le.at[i,'wd']]
    for i in range(0,len(le.at[i,'dw'])):
        i_str='%02d' %i
        le.loc[:,'dw'+i_str]=le.loc[:,'dw'].map(lambda x: x[i])
        le.loc[:,'nw'+i_str]=le.loc[:,'nw'].map(lambda x: x[i])
    for i in range(0,len(le.at[i,'wd'])):
        i_str='%02d' %i
        le.loc[:,'wd'+i_str]=le.loc[:,'wd'].map(lambda x: x[i])
    le.drop(columns=['dw','nw','wd'],inplace=True)

    splitpoint=int(len(op)/output_step*ratio)
    data=st.to_numpy().reshape((int(len(st)/fore_step),fore_step,sta_num,feature))/60.0
    st_train=data[:splitpoint]
    st_test=data[splitpoint:]
    print('short term',data.shape)
    #print(data[0])

    data=lt.to_numpy().reshape((int(len(lt)/fore_step),fore_step,sta_num,feature))/60.0
    lt_train=data[:splitpoint]
    lt_test=data[splitpoint:]
    print('long term',data.shape)
    #print(data[0])

    se.loc[:,'AQI']=np.ceil(se.loc[:,'AQI']/50.0)
    data=se.to_numpy()
    se_train=data[:splitpoint]
    se_test=data[splitpoint:]
    print('short external feature',data.shape)
    #print(data[0])

    le.loc[:,'AQI']=np.ceil(le.loc[:,'AQI']/50.0)
    data=le.to_numpy()
    data=data.reshape((int(len(le)/fore_step),fore_step,len(le.columns)))
    le_train=data[:splitpoint]
    le_test=data[splitpoint:]
    print('long external feature',data.shape)
    #print(data[0])    

    data=op.to_numpy()/60.0
    op_train=data[:splitpoint]
    op_test=data[splitpoint:]
    print('output',data.shape)
    #print(data[0])

    return st_train,st_test,lt_train,lt_test,se_train,se_test,le_train,le_test,op_train,op_test

def visualize_train(prefix:str,history):

    pyplot.figure(figsize=(16,9))
    pyplot.title('history metrics(lower is better)')
    pyplot.xlabel('epoch')
    pyplot.ylabel('loss(rmse)/mae(100s)/accuracy')
    pyplot.ylim(ymax=1.0)
    
    for i in range(0,len(history.history['loss'])):
        #turn mse to rmse
        history.history['loss'][i]=np.sqrt(history.history['loss'][i])
        history.history['val_loss'][i]=np.sqrt(history.history['val_loss'][i])
        #turn mae in min to mape in 100sec
        history.history['mae'][i]=history.history['mae'][i]*0.6
        history.history['val_mae'][i]=history.history['val_mae'][i]*0.6
        #turn mape to accuracy
        history.history['mape'][i]=1-history.history['mape'][i]/100.0
        history.history['val_mape'][i]=1-history.history['val_mape'][i]/100.0

    pyplot.plot(history.history['loss'],label='loss',color='r',linestyle=':',linewidth=1.0)
    pyplot.plot(history.history['val_loss'],label='val_loss',color='r',linestyle='-',linewidth=1.0)

    pyplot.plot(history.history['mae'],label='mae',color='b',linestyle=':',linewidth=1.0)
    pyplot.plot(history.history['val_mae'],label='val_mae',color='b',linestyle='-',linewidth=1.0)
    
    pyplot.plot(history.history['mape'],label='accuracy',color='g',linestyle=':',linewidth=1.0)
    pyplot.plot(history.history['val_mape'],label='val_accuracy',color='g',linestyle='-',linewidth=1.0)
    
    pyplot.legend()
    pyplot.savefig(prefix+'history.png')

def train_model(
                prefix:str,
                model:Model,
                x:np.ndarray,
                y:np.ndarray,
                xtest:np.ndarray,
                ytest:np.ndarray,
                batch_size=16,
                epochs=2000,
                verbose=1
                )->Model:
    prefix=prefix.split('/')
    dl=os.listdir()#pwd: ${ori}
    if prefix[0] not in dl:
        os.mkdir(prefix[0])
    os.chdir(prefix[0])# pwd: temp/
    dl=os.listdir()
    if prefix[1] not in dl:
        os.mkdir(prefix[1])
    os.chdir('..')#pwd: ${ori}
    prefix=prefix[0]+'/'+prefix[1]
    
    prefix='./'+prefix+'/'#./temp/${model_name}/
    path='model_{epoch:04d}_{val_loss:.04f}_{val_mae:.04f}_{val_mape:04f}.hdf5'
    checkpoint=CustomModelCheckpoint(dirpath=prefix,filepath=path,period=10)
    earlystop=EarlyStopping(patience=20,restore_best_weights=True)
    reduce_lr=ReduceLROnPlateau(patience=5)
    callback_list=[checkpoint,earlystop,reduce_lr]
    
    clock=datetime.datetime.now()
    history=model.fit(x,y,batch_size=16,epochs=epochs,callbacks=callback_list,validation_data=[xtest,ytest],verbose=verbose)
    clock=datetime.datetime.now()-clock

    print('time consumption:',clock)
    save_model(model,prefix+'best_model.hdf5')
    model.save_weights(prefix+'best_weight.hdf5')
    visualize_train(prefix,history)
    return model

def visualize_test(prefix:str,y_true,y_pred):#plot last day prediction
    prefix='./'+prefix+'/'
    pyplot.figure(figsize=(21,9))
    pyplot.title('pred vs. true')
    if len(y_true)>20*72:#just plot one sta of last day of pred data sipailou downflow sta_order=15
        y_true=y_true[-20*72+14::20]
        y_pred=y_pred[-20*72+14::20]
    pyplot.plot(y_true,label='y_true',linewidth=1)
    pyplot.plot(y_pred,label='y_pred',linewidth=1,linestyle='-')
    pyplot.legend()
    pyplot.show()
    pyplot.savefig(prefix+'y_pred.png')
