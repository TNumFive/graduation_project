from utility import ConfigStruct,train_model
from model import example_model,convlstm
from prepare_data import prepare_data_tt
from matplotlib import pyplot
import pandas as pd
import numpy as np
import os
import sys
np.random.seed(seed=5)
#pf:save to: m_0059_9.4761_2.0833_368238340.6299.hdf5
if __name__ == "__main__":

    #script will create dir called ./temp/${prefix} to store model,weight history and pred vs. true
    prefix=os.path.basename(sys.argv[0])
    prefix=prefix[:-3]
    #config is a struct contains all settings 
    config=ConfigStruct()
    config.model_name=prefix
    #create model with settings from cofig
    model=convlstm(config)
    #generate data 
    x_train,x_test,y_train,y_test=prepare_data_tt()
    #train model
    model=train_model(config,model,x_train,y_train,x_test,y_test)
    #visualize predict
    y_pred=model.predict(x_test)
    print(y_pred.shape)

