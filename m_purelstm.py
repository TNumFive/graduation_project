from utility import ConfigStruct,train_model
from model import example_model,purelstm
from prepare_data import prepare_data_pf
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
    config.model_name=prefix+'_pf'
    #create model with settings from cofig
    model=purelstm(config)
    #generate data 
    x_train,x_test,y_train,y_test=prepare_data_pf()
    #train model
    model=train_model(config,model,x_train,y_train,x_test,y_test)
    #visualize predict
    y_pred=model.predict(x_test)
    print(y_pred.shape)

