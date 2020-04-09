from utility import custom_scale2,train_model,visualize_test
from matplotlib import pyplot
import numpy as np
import os

np.random.seed(seed=5)

if __name__ == "__main__":

    prefix='lst_conv1d_lstm'#script will create dir called ./temp/${prefix} to store model,weight history and pred vs. true
    prefix='temp/'+prefix

    #generate data
    st_train,st_test,lt_train,lt_test,se_train,se_test,le_train,le_test,op_train,op_test=custom_scale2()
    
    from keras.models import Sequential,load_model,Model
    from keras.layers import BatchNormalization,LSTM,Dropout,Dense,RepeatVector,Flatten,TimeDistributed,concatenate,Input,Conv1D
    from keras import layers

    #build model here
    ip1=Input(shape=(4,22,1))
    ip2=Input(shape=(4,22,1))
    
    op1=TimeDistributed(Conv1D(filters=1,kernel_size=6,padding='same'))(ip1)
    op2=TimeDistributed(Conv1D(filters=1,kernel_size=6,padding='same'))(ip2)
    op1=layers.Reshape((4,22))(op1)
    op2=layers.Reshape((4,22))(op2)
    op=concatenate([op1,op2])
    
    op=BatchNormalization()(op)
    op=LSTM(64,return_sequences=True)(op)
    op=Dropout(0.2)(op)
    op=BatchNormalization()(op)
    op=LSTM(64,return_sequences=False)(op)
    op=Dropout(0.1)(op)

    op=RepeatVector(1)(op)
    op=BatchNormalization()(op)
    op=LSTM(64,return_sequences=True)(op)
    op=Dropout(0.1)(op)
    op=BatchNormalization()(op)
    op=LSTM(64,return_sequences=True)(op)
    op=TimeDistributed(Dense(22,activation='relu'))(op)
    op=Flatten()(op)
    model=Model(inputs=[ip1,ip2],outputs=op)
    #compile model here,use default mae mape and mse ,when compare we can compute base on mse to get rmse
    model.compile(optimizer='adam',metrics=['mae','mape'],loss='mse')
    model.summary()

    #final process on preprocess data
    
    #assign train:x,y;  test:x,y
    x_train=[st_train,lt_train]
    x_test=[st_test,lt_test]
    y_train=op_train
    y_test=op_test
    #train model with custom-modelcheckpoint, earylstopping and reducelronplateau callback
    model=train_model(prefix,model,x_train,y_train,x_test,y_test,verbose=1)
    
    #load model from file or use the model trained by above function to predict
    #model=load_model('./'+prefix+'/'+'best_model.hdf5')
    y_pred=model.predict(x_test)
    #use flatten data in case 1000 lines on a graph
    visualize_test(prefix,op_test.flatten(),y_pred.flatten())