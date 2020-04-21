from utility import custom_scale2,custom_scale3,train_model,visualize_test
from matplotlib import pyplot
import pandas as pd
import numpy as np
import os
import sys
np.random.seed(seed=5)

def prepare_data(forestep=32):
    
    file_list=os.listdir('./data')
    if 'tt_dataset_multi_output.csv' in file_list:
        dataset:pd.DataFrame=pd.read_csv('./data/tt_dataset_multi_output.csv',parse_dates=['start_time'])
    else:
        dataset=pd.DataFrame()
        ttdataset=pd.read_csv('./data/tt_dataset.csv',parse_dates=['start_time'])
        ttdataset.set_index(['start_time'],inplace=True,drop=False)
        len_ttdataset=len(ttdataset)
        for i in range(33,len_ttdataset):
            start:pd.Timestamp=ttdataset.iloc[i-33].name
            now:pd.Timestamp=ttdataset.iloc[i].name
            if now.day-start.day<=1:
                dataset=dataset.append(ttdataset.iloc[i-33:i],ignore_index=True)
            print('\r\tprepare data: ',i+1,'/',len_ttdataset,sep='',end='')
        print('')
        dataset.to_csv('./data/tt_dataset_multi_output.csv',index=False)
    dataset.set_index(['start_time'],inplace=True)
    dataset=dataset.to_numpy().reshape(int(len(dataset)/33),33,len(dataset.columns))
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

if __name__ == "__main__":

    #script will create dir called ./temp/${prefix} to store model,weight history and pred vs. true
    prefix=os.path.basename(sys.argv[0])
    prefix=prefix[:-3]
    prefix='temp/'+prefix

    #generate data
    x_train,x_test,y_train,y_test=prepare_data()
    
    from keras.models import Sequential,load_model,Model
    from keras import layers
    from keras.layers import BatchNormalization,LSTM,Dropout,Dense,RepeatVector,Flatten,TimeDistributed,concatenate,Input

    #build model here
    model=Sequential()
    model.add(BatchNormalization(input_shape=(4,22)))
    model.add(TimeDistributed(Dense(units=64)))
    model.add(BatchNormalization())
    model.add(TimeDistributed(Dense(units=64)))
    model.add(Flatten())
    model.add(RepeatVector(1))
    model.add(TimeDistributed(Dense(64)))
    model.add(BatchNormalization())
    model.add(TimeDistributed(Dense(22,activation='linear')))
    model.add(Flatten())
    #compile model here,use default mae mape and mse ,when compare we can compute base on mse to get rmse
    model.compile(optimizer='adam',metrics=['mae','mape'],loss='mse')
    model.summary()

    #final process on preprocess data
    st_train=st_train.squeeze()
    st_test=st_test.squeeze()
    #assign train:x,y;  test:x,y
   
    #train model with custom-modelcheckpoint, earylstopping and reducelronplateau callback
    model=train_model(prefix,model,x_train,y_train,x_test,y_test,verbose=1)
    
    #load model from file or use the model trained by above function to predict
    #model=load_model('./'+prefix+'/'+'best_model.hdf5')
    #   please compile the model and then load weight ,as it seems to be some bugs with the load model functions when custom layers existed
    #model.load_weights('./'+prefix+'/best_weight.hdf5')
    y_pred=model.predict(x_test)
    #use flatten data in case 1000 lines on a graph
    visualize_test(prefix,y_test.flatten(),y_pred.flatten())