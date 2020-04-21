from utility import custom_scale2,custom_scale3,train_model,visualize_test
from matplotlib import pyplot
import pandas as pd
import numpy as np
import os
import sys
np.random.seed(seed=5)

def prepare_data(forestep=7,foreday=3):
    file_list=os.listdir('./data')
    if 'tt_dataset_stdn.csv' in file_list:
        dataset:pd.DataFrame=pd.read_csv('./data/tt_dataset_stdn.csv',parse_dates=['start_time'])
    else:
        dataset=pd.DataFrame()
        ttdataset=pd.read_csv('./data/tt_dataset.csv',parse_dates=['start_time'])
        ttdataset.set_index(['start_time'],inplace=True,drop=False)
        len_ttdataset=len(ttdataset)
        for i in range(foreday*72+1+forestep,len_ttdataset):
            now:pd.Timestamp=ttdataset.iloc[i].name
            start:pd.Timestamp=ttdataset.iloc[i-72*foreday].name
            if now.day-start.day==3:
                for j in range(-1*foreday,1):
                    dataset=dataset.append(ttdataset.iloc[i+72*j-forestep:i+72*j+1])
            print('\r\tprepare data: ',i+1,'/',len_ttdataset,sep='',end='')
        print('')
        dataset.to_csv('./data/tt_dataset_stdn.csv',index=False)
    dataset.set_index(['start_time'],inplace=True,drop=True)
    dataset=dataset.to_numpy().reshape(int(len(dataset)/32),32,len(dataset.columns))
    dataset=dataset/60#turn seconds to mins to decrease calculation
    print('dataset.shape:',dataset.shape)
    splitratio=int(0.7*dataset.shape[0])
    train=dataset[:splitratio]
    l_train=train[:,:-8]
    s_train=train[:,-8:-4]
    y_train=train[:,-4]
    test=dataset[splitratio:]
    l_test=test[:,:-8]
    s_test=test[:,-8:-4]
    y_test=test[:,-4]
    print('l_train.shape:',l_train.shape)
    print('s_train.shape:',s_train.shape)
    print('y_train.shape:',y_train.shape)
    
    return l_train,l_test,s_train,s_test,y_train,y_test

if __name__ == "__main__":
    
    #script will create dir called ./temp/${prefix} to store model,weight history and pred vs. true
    prefix=os.path.basename(sys.argv[0])
    prefix=prefix[:-3]
    prefix='temp/'+prefix

    #generate data
    #st_train,st_test,lt_train,lt_test,se_train,se_test,le_train,le_test,op_train,op_test=custom_scale3()
    l_train,l_test,s_train,s_test,y_train,y_test=prepare_data()

    from keras.models import Sequential,load_model,Model
    from keras import layers
    from keras.layers import BatchNormalization,LSTM,Dropout,Dense,RepeatVector,Flatten,TimeDistributed,concatenate,Input
    from attention import Attention
    #build model here
    lt=Input(shape=(24,20))
    st=Input(shape=(4,20))
    op1=layers.Reshape((24,20,1))(lt)
    op1=TimeDistributed(layers.Conv1D(filters=64,kernel_size=6,padding='same'))(op1)
    op1=layers.Reshape((3,8,20*64))(op1)
    op1=TimeDistributed(LSTM(64,return_sequences=True))(op1)
    op1=TimeDistributed(LSTM(64,return_sequences=True))(op1)
    op1=TimeDistributed(Attention())(op1)
    op1=Flatten()(op1)  

    op2=layers.Reshape((4,20,1))(st)
    op2=TimeDistributed(layers.Conv1D(filters=64,kernel_size=6,padding='same'))(op2)
    op2=layers.Reshape((4,20*64))(op2)
    op2=LSTM(64,return_sequences=True)(op2)
    op2=LSTM(64,return_sequences=True)(op2)
    op2=Attention()(op2)
    op=concatenate([op1,op2])
    op=Dense(20,activation='relu')(op)
    model=Model(inputs=[lt,st],outputs=[op],name='STDN-2')
    #compile model here,use default mae mape and mse ,when compare we can compute base on mse to get rmse
    model.compile(optimizer='adam',metrics=['mae','mape'],loss='mse')
    model.summary()


    #final process on preprocess data

    #assign train:x,y;  test:x,y
    x_train=[l_train,s_train]
    x_test=[l_test,s_test]
    y_train=y_train
    y_test=y_test

    #train model with custom-modelcheckpoint, earylstopping and reducelronplateau callback
    model=train_model(prefix,model,x_train,y_train,x_test,y_test,verbose=1)
    
    #load model from file or use the model trained by above function to predict
    #   please compile the model and then load weight ,as it seems to be some bugs with the load model functions
    #model.load_weights('./'+prefix+'/best_weight.hdf5')
    y_pred=model.predict(x_test)
    #use flatten data in case 1000 lines on a graph
    visualize_test(prefix,y_test.flatten(),y_pred.flatten())
