from utility import custom_scale2,train_model
from matplotlib import pyplot
import numpy as np
import os

np.random.seed(seed=5)

if __name__ == "__main__":
    st_train,st_test,lt_train,lt_test,se_train,se_test,le_train,le_test,op_train,op_test=custom_scale2()
    
    from keras.models import Sequential,Model
    from keras.layers import BatchNormalization,LSTM,Dropout,Dense,RepeatVector,Conv1D,concatenate,Input

    #build long short term lstm model
    ip1=Input(shape=(4,22))
    ip2=Input(shape=(4,22))

    op1=BatchNormalization()(ip1)
    op1=LSTM(64,return_sequences=True)(op1)
    op1=Dropout(0.2)(op1)
    op1=BatchNormalization()(op1)
    op1=LSTM(64,return_sequences=False)(op1)
    op1=Dropout(0.1)(op1)
    
    op2=BatchNormalization()(ip2)
    op2=LSTM(64,return_sequences=True)(op2)
    op2=Dropout(0.2)(op2)
    op2=BatchNormalization()(op2)
    op2=LSTM(64,return_sequences=False)(op2)
    op2=Dropout(0.1)(op2)
    
    op=concatenate([op1,op2])
    op=BatchNormalization()(op)
    op=RepeatVector(1)(op)
    op=LSTM(64,return_sequences=True)(op)
    op=Dropout(0.1)(op)
    
    op=BatchNormalization()(op)
    op=LSTM(64,return_sequences=False)(op)
    op=Dense(22,activation='linear')(op)
    model=Model(inputs=[ip1,ip2],outputs=[op])

    #compile model
    model.compile(optimizer='adam',metrics=['mae','mape'],loss='mse')
    model.summary()

    prefix='lstlstm'
    prefix='temp/'+prefix
    st_train=st_train.squeeze()
    st_test=st_test.squeeze()
    lt_train=lt_train.squeeze()
    lt_test=lt_test.squeeze()

    model=train_model(prefix,model,[st_train,lt_train],op_train,[st_test,lt_test],op_test,verbose=1)
    prefix='./'+prefix+'/'
    #model=load_model(prefix+'best_model.hdf5')
    y_pred=model.predict([st_test,lt_test])
    
    y_pred=y_pred.flatten()
    op_test=op_test.flatten()
    pyplot.figure(figsize=(21,9))
    pyplot.title('pred vs. true')
    pyplot.plot(y_pred,label='y_pred',linewidth=0.2)
    pyplot.plot(op_test,label='y_true',linewidth=0.2)
    pyplot.legend()
    pyplot.show()
    pyplot.savefig(prefix+'y_pred.png')
