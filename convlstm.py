from utility import custom_scale2,train_model,mae,mape
from matplotlib import pyplot
import numpy as np
import os

np.random.seed(seed=5)

if __name__ == "__main__":
    st_train,st_test,lt_train,lt_test,se_train,se_test,le_train,le_test,op_train,op_test=custom_scale2()
    
    from keras.models import Sequential,load_model
    from keras.layers import BatchNormalization,LSTM,Dropout,Dense,ConvLSTM2D,RepeatVector,Flatten,Reshape,TimeDistributed

    #build model
    model=Sequential()
    model.add(BatchNormalization(name='bn0',input_shape=(4,22,1,1)))
    model.add(ConvLSTM2D(name='convlstm1',filters=64,kernel_size=(10,1),padding='same',return_sequences=True))
    model.add(Dropout(0.2,name='dropout1'))
    model.add(BatchNormalization(name='bn1'))
    model.add(ConvLSTM2D(name='convlstm2',filters=64,kernel_size=(5,1),padding='same',return_sequences=False))
    model.add(Dropout(0.1,name='dropout2'))
    model.add(BatchNormalization(name='bn2'))
    model.add(Flatten())
    model.add(RepeatVector(1))
    model.add(Reshape((1,22,1,64)))
    model.add(ConvLSTM2D(name='convlstm3',filters=64,kernel_size=(10,1),padding='same',return_sequences=True))
    model.add(Dropout(0.1,name='dropout3'))
    model.add(BatchNormalization(name='bn3'))
    model.add(ConvLSTM2D(name='convlstm4',filters=64,kernel_size=(5,1),padding='same',return_sequences=True))
    model.add(TimeDistributed(Dense(units=1,name='dense1',activation='relu')))
    model.add(Flatten())
    #compile model
    model.compile(optimizer='adam',metrics=['mae','mape'],loss='mse')
    model.summary()

    prefix='convlstm'
    prefix='temp/'+prefix
    st_train=st_train[:,:,:,:,np.newaxis]
    st_test=st_test[:,:,:,:,np.newaxis]
    
    sw1=sw2=True

    if sw1:
        model=train_model(prefix,model,st_train,op_train,st_test,op_test,verbose=1)
    if sw2:
        prefix='./'+prefix+'/'
        #model=load_model(prefix+'best_model.hdf5')
        y_pred=model.predict(st_test)
    
        y_pred=y_pred.flatten()
        op_test=op_test.flatten()
        pyplot.figure(figsize=(21,9))
        pyplot.title('pred vs. true')
        pyplot.plot(op_test,label='y_true',linewidth=0.2)
        pyplot.plot(y_pred,label='y_pred',linewidth=0.2)
        pyplot.legend()
        pyplot.show()
        pyplot.savefig(prefix+'y_pred.png')
