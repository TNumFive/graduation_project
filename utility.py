import numpy as np
np.random.seed(seed=5)

import pandas as pd
import os 
import datetime
from matplotlib import pyplot
from keras import Model
from keras.callbacks import Callback,EarlyStopping,ReduceLROnPlateau
from keras.models import save_model,load_model
from keras import backend as K
from sklearn.preprocessing import OneHotEncoder

class ConfigStruct:
    def __init__(self):
        self.model_name='example_model'
        self.directory='temp'
        self.namestyle='m_{epoch:04d}_{val_loss:.04f}_{val_mae:.04f}_{val_mape:.04f}.hdf5'
        self.input_timestep=32
        self.sta_num=20
        self.output_timestep=1
        self.optimizer='adam'
        self.metrics=['mae','mape']
        self.loss='mse'
        self.custom=dict()

def print_log(s:str,end='\n',log_name='runtime.log'):
        runtime=open(log_name,'a+')
        runtime.write(s+end)
        runtime.close()

class CustomModelCheckpoint(Callback):
    def __init__(self, directory, namestyle, monitor='val_loss', 
                 mode='auto', period=1,verbose=1):
        super(CustomModelCheckpoint, self).__init__()
        self.monitor = monitor
        self.directory = directory
        self.namestyle=namestyle
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
            self.model.save(self.directory+'/model.hdf5')
            self.best_epoch=epoch
            self.best_weight=self.model.get_weights()
            for k in logs.keys():
                self.best_logs[k]=logs[k]
        if self.epochs_since_last_save>=self.period:#time to save 
            self.epochs_since_last_save=0
            if self.update==1:
                self.update=0
                path=self.namestyle.format(epoch=self.best_epoch+1,**(self.best_logs))
                os.rename(self.directory+'/model.hdf5',self.directory+'/'+path)
                if self.verbose==1:
                    print('save to:',path)
            if epoch!=self.best_epoch:#如果当前epoch不是最佳epoch，就把最佳权重更新上去，保证新的一轮以最佳权重为基础
                self.model.set_weights(self.best_weight)
    
    def on_train_end(self,logs=None):
        if self.update==1:#updated but not saved，basically impossible,but just in case
            path=self.namestyle.format(epoch=self.best_epoch+1,**(self.best_logs))
            os.rename(self.directory+'/model.hdf5',self.directory+'/'+path)
            if self.verbose==1:
                print('save to:',path)
        print_log('')#start a new line
        print_log('Timestamp:'+str(datetime.datetime.now()))#模型保存时间
        self.model.summary(print_fn=print_log)#模型结构
        print_log(self.namestyle.format(epoch=self.best_epoch+1,**(self.best_logs)))#模型运行结果

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

def train_model(config:ConfigStruct,
                model:Model,
                x_train:np.ndarray,
                y_train:np.ndarray,
                x_test:np.ndarray,
                y_test:np.ndarray,
                epochs=2000,
                batch_size=16,
                verbose=1)->Model:
    dl=os.listdir('./')#pwd
    if config.directory not in dl:
        os.mkdir(config.directory)
    dl=os.listdir('./'+config.directory)
    prefix='./'+config.directory+'/'+config.model_name
    if config.model_name not in dl:
        os.mkdir(prefix)
    path=config.namestyle
    checkpoint=CustomModelCheckpoint(prefix,path,period=10)
    earlystop=EarlyStopping(patience=20,restore_best_weights=True)
    reduce_lr=ReduceLROnPlateau(patience=5)
    callback_list=[checkpoint,earlystop,reduce_lr]
    
    clock=datetime.datetime.now()
    history=model.fit(x_train,y_train,batch_size=batch_size,epochs=epochs,verbose=verbose,callbacks=callback_list,validation_data=[x_test,y_test])
    clock=datetime.datetime.now()-clock
    model.save(prefix+'/best_model.hdf5')
    model.save_weights(prefix+'/best_weight.hdf5')
    print('time consumption:',clock)
    print_log('time consumption:%s' %(str(clock)))
    
    pyplot.figure(figsize=(16,9))
    pyplot.title('history metrics')
    pyplot.xlabel('epoch')
    #pyplot.ylim(ymin=0.0)
    pyplot.plot(history.history['loss'],label='loss',color='r',linestyle=':',linewidth=1.0)
    pyplot.plot(history.history['val_loss'],label='val_loss',color='r',linestyle='-',linewidth=1.0)
    pyplot.legend()
    pyplot.savefig(prefix+'/history.png')
    
    return model
