from utility import custom_scale2,train_model,visualize_test
from matplotlib import pyplot
import numpy as np
import os

np.random.seed(seed=5)

if __name__ == "__main__":

    prefix='puredense'#script will create dir called ./temp/${prefix} to store model,weight history and pred vs. true
    prefix='temp/'+prefix

    #generate data
    st_train,st_test,lt_train,lt_test,se_train,se_test,le_train,le_test,op_train,op_test=custom_scale2()
    
    from keras.models import Sequential,load_model,Model
    from keras.layers import BatchNormalization,LSTM,Dropout,Dense,RepeatVector,Flatten,TimeDistributed,concatenate,Input,ConvLSTM2D
    from keras.layers import Reshape
    from keras import layers as kl

    #build model
    ip1=Input(shape=(4,22,1,1))
    ip2=Input(shape=(4,22,1,1))
    ip=concatenate([ip1,ip2])
    op=BatchNormalization()(ip)
    op=ConvLSTM2D(filters=64,kernel_size=(10,1),padding='same',return_sequences=True)(op)
    op=Dropout(0.2)(op)
    op=BatchNormalization()(op)
    op=ConvLSTM2D(filters=64,kernel_size=(5,1),padding='same',return_sequences=False)(op)
    op=Dropout(0.1)(op)
    op=BatchNormalization()(op)
    op=Flatten()(op)
    op=RepeatVector(1)(op)
    op=Reshape((1,22,1,64))(op)
    op=ConvLSTM2D(filters=64,kernel_size=64,padding='same',return_sequences=True)(op)
    op=Dropout(0.1)(op)
    op=BatchNormalization()(op)
    op=ConvLSTM2D(filters=64,kernel_size=(5,1),padding='same',return_sequences=True)(op)
    op=TimeDistributed(Dense(1,activation='relu'))(op)
    op=Flatten()(op)
    model=Model(inputs=[ip1,ip2],outputs=op)
    #compile model here,use default mae mape and mse ,when compare we can compute base on mse to get rmse
    model.compile(optimizer='adam',metrics=['mae','mape'],loss='mse')
    model.summary()

    #final process on preprocess data
    st_train=st_train[:,:,:,:,np.newaxis]
    st_test=st_test[:,:,:,:,np.newaxis]
    lt_train=lt_train[:,:,:,:,np.newaxis]
    lt_test=lt_test[:,:,:,:,np.newaxis]
    #assign train:x,y;  test:x,y
    x_train=[st_train,lt_train]
    x_test=[st_test,lt_test]
    y_train=op_train
    y_test=op_test
    #train model with custom-modelcheckpoint, earylstopping and reducelronplateau callback
    model=train_model(prefix,model,x_train,y_train,x_test,y_test,verbose=1)
    
    #load model from file or use the model trained by above function to predict
    model=load_model('./'+prefix+'/'+'best_model.hdf5')
    y_pred=model.predict(x_test)
    #use flatten data in case 1000 lines on a graph
    visualize_test(prefix,op_test.flatten(),y_pred.flatten())