from utility import ConfigStruct,train_model
from model import example_model,convlstm
from prepare_data import prepare_data_ttpf
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot
import pandas as pd
import numpy as np
import os
import sys
np.random.seed(seed=5)

if __name__ == "__main__":

    #script will create dir called ./temp/${prefix} to store model,weight history and pred vs. true
    prefix=os.path.basename(sys.argv[0])
    prefix=prefix[:-3]
    #config is a struct contains all settings 
    config=ConfigStruct()
    config.model_name=prefix
    config.input_timestep=8
    #create model with settings from cofig
    model=convlstm(config)
    #generate data 
    data=prepare_data_ttpf()
    weekday=data['weekday']
    pf_train=data['pf_train']
    pf_test=data['pf_test']
    pfmms:MinMaxScaler=data['pfmms']
    x_train=pf_train[:,-9:-1]
    y_train=pf_train[:,-1]
    w_train=weekday
    x_test=pf_test[:,-9:-1]
    y_test=pf_test[:,-1]
    '''
    #train model
    model=train_model(config,model,x_train,y_train,x_test,y_test)
    #visualize predict
    y_pred=model.predict(x_test)
    y_pred=pfmms.inverse_transform(y_pred)
    y_test=pfmms.inverse_transform(y_test)
    from sklearn.metrics import mean_absolute_error,mean_squared_error
    print('mae:',mean_absolute_error(y_test,y_pred))
    print('mse:',mean_squared_error(y_test,y_pred))
    '''