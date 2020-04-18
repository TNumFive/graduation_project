from utility import custom_scale2,custom_scale3,train_model,visualize_test
from matplotlib import pyplot
import numpy as np
import os
import sys
np.random.seed(seed=5)

if __name__ == "__main__":
    
    #script will create dir called ./temp/${prefix} to store model,weight history and pred vs. true
    prefix=os.path.basename(sys.argv[0])
    prefix=prefix[:-3]
    prefix='temp/'+prefix

    #generate data
    st_train,st_test,lt_train,lt_test,se_train,se_test,le_train,le_test,op_train,op_test=custom_scale3()
    
    from keras.models import Sequential,load_model,Model
    from keras import layers
    from keras.layers import BatchNormalization,LSTM,Dropout,Dense,RepeatVector,Flatten,TimeDistributed,concatenate,Input
    from attention import Attention
    #build model here
    lt=Input(shape=(4,22,1))
    st=Input(shape=(4,22,1))
    le=Input(shape=(4,le_train.shape[2]))
    se=Input(shape=(se_train.shape[1],))
    
    op1=TimeDistributed(layers.Conv1D(filters=1,kernel_size=6,padding='same'))(lt)
    op1=layers.Reshape((4,22))(op1)
    #op1=concatenate([op1,le])
    op1=LSTM(units=64,return_sequences=True)(op1)
    op1=Attention(method='cba')(op1)

    op2=TimeDistributed(layers.Conv1D(filters=1,kernel_size=6,padding='same'))(st)
    op2=layers.Reshape((4,22))(op2)
    op2=LSTM(units=64,return_sequences=True)(op2)
    op2=Attention(method='cba')(op2)
    #op2=concatenate([op2,se])

    op=concatenate([op1,op2])
    op=RepeatVector(1)(op)
    op=TimeDistributed(Dense(22))(op)
    op=TimeDistributed(layers.Activation('relu'))(op)
    op=Flatten()(op)
    model=Model(inputs=[lt,st,le,se],outputs=[op])
    #compile model here,use default mae mape and mse ,when compare we can compute base on mse to get rmse
    model.compile(optimizer='adam',metrics=['mae','mape'],loss='mse')
    model.summary()

    #final process on preprocess data

    #assign train:x,y;  test:x,y
    x_train=[lt_train,st_train,le_train,se_train]
    x_test=[lt_test,st_test,le_test,se_test]
    y_train=op_train
    y_test=op_test

    #train model with custom-modelcheckpoint, earylstopping and reducelronplateau callback
    model=train_model(prefix,model,x_train,y_train,x_test,y_test,verbose=1)
    
    #load model from file or use the model trained by above function to predict
    #   please compile the model and then load weight ,as it seems to be some bugs with the load model functions
    #model.load_weights('./'+prefix+'/best_weight.hdf5')
    y_pred=model.predict(x_test)
    #use flatten data in case 1000 lines on a graph
    visualize_test(prefix,op_test.flatten(),y_pred.flatten())