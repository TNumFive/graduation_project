from utility import custom_scale2,train_model,visualize_test
from matplotlib import pyplot
import numpy as np
import os

np.random.seed(seed=5)

if __name__ == "__main__":

    prefix='pure_dense'#script will create dir called ./temp/${prefix} to store model,weight history and pred vs. true
    prefix='temp/'+prefix

    #generate data
    st_train,st_test,lt_train,lt_test,se_train,se_test,le_train,le_test,op_train,op_test=custom_scale2()
    
    from keras.models import Sequential,load_model,Model
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
    x_train=st_train
    x_test=st_test
    y_train=op_train
    y_test=op_test
    #train model with custom-modelcheckpoint, earylstopping and reducelronplateau callback
    model=train_model(prefix,model,x_train,y_train,x_test,y_test,verbose=1)
    
    #load model from file or use the model trained by above function to predict
    #model=load_model('./'+prefix+'/'+'best_model.hdf5')
    y_pred=model.predict(x_test)
    #use flatten data in case 1000 lines on a graph
    visualize_test(prefix,op_test.flatten(),y_pred.flatten())