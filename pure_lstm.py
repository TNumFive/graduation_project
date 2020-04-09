from utility import custom_scale2,train_model,visualize_test
from matplotlib import pyplot
import numpy as np
import os

np.random.seed(seed=5)

if __name__ == "__main__":
    prefix='pure_lstm'#script will create dir called ./temp/${prefix} to store model,weight history and pred vs. true
    prefix='temp/'+prefix

    #generate data
    st_train,st_test,lt_train,lt_test,se_train,se_test,le_train,le_test,op_train,op_test=custom_scale2()

    from keras.models import Sequential,load_model
    from keras.layers import BatchNormalization,LSTM,Dropout,Dense,RepeatVector,TimeDistributed,Flatten

    #build model
    model=Sequential()
    model.add(BatchNormalization(name='bn0',input_shape=(4,22)))
    model.add(LSTM(name='lstm1',units=64,return_sequences=True))
    model.add(Dropout(0.2,name='dropout1'))
    model.add(BatchNormalization(name='bn1'))
    model.add(LSTM(name='lstm2',units=64,return_sequences=False))
    model.add(Dropout(0.1,name='dropout2'))
    model.add(BatchNormalization(name='bn2'))
    model.add(RepeatVector(1))
    model.add(LSTM(name='lstm3',units=64,return_sequences=True))
    model.add(Dropout(0.1,name='dropout3'))
    model.add(BatchNormalization(name='bn3'))
    model.add(LSTM(name='lstm4',units=64,return_sequences=True))
    model.add(TimeDistributed(Dense(name='dense1',units=22,activation='linear')))
    model.add(Flatten())
    #compile model
    model.compile(optimizer='adam',metrics=['mae','mape'],loss='mse')
    model.summary()

    st_train=st_train.squeeze()
    st_test=st_test.squeeze()

    model=train_model(prefix,model,st_train,op_train,st_test,op_test,verbose=1)
    #model=load_model('./'+prefix+'/'+'best_model.hdf5')
    y_pred=model.predict(st_test)
    visualize_test(prefix,op_test.flatten(),y_pred.flatten())