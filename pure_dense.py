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
    
    from keras.models import Sequential,load_model
    from keras.layers import BatchNormalization,LSTM,Dropout,Dense,RepeatVector,Flatten,TimeDistributed

    #build model
    #if use relu:save to:model_0013_0.2626_0.3480_19.804995.hdf5
    #if not use relu: save to:model_0012_0.2497_0.3377_19.049406.hdf5
    #with reduce_lr:save to: model_0022_0.2470_0.3356_19.113960.hdf5
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
    #compile model
    model.compile(optimizer='adam',metrics=['mae','mape'],loss='mse')
    model.summary()

    st_train=st_train.squeeze()
    st_test=st_test.squeeze()

    model=train_model(prefix,model,st_train,op_train,st_test,op_test,verbose=1)
    model=load_model('./'+prefix+'/'+'best_model.hdf5')
    y_pred=model.predict(st_test)
    visualize_test(prefix,op_test.flatten(),y_pred.flatten())