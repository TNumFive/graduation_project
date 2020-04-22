from keras import layers
from keras import Model
from keras import models
from keras import backend as K

from utility import ConfigStruct

def example_model(config:ConfigStruct,summary=True)->Model:
    #import what you need
    from keras.layers import BatchNormalization,LSTM,Dropout,RepeatVector,TimeDistributed,Dense,Flatten

    #get settings from config struct
    input_timestep=config.input_timestep
    sta_num=config.sta_num
    output_timestep=config.output_timestep

    #start build model
    model=models.Sequential(name=config.name)
    model.add(BatchNormalization(name = 'batch_norm_0', input_shape = (input_timestep,sta_num)))
    model.add(LSTM(name ='lstm_1',
                   units = 64,
                   return_sequences = True))
    
    model.add(Dropout(0.2, name = 'dropout_1'))
    model.add(BatchNormalization(name = 'batch_norm_1'))

    model.add(LSTM(name ='lstm_2',
                   units = 64,
                   return_sequences = False))
    
    model.add(Dropout(0.1, name = 'dropout_2'))
    model.add(BatchNormalization(name = 'batch_norm_2'))
    
    model.add(RepeatVector(output_timestep))
    
    model.add(LSTM(name ='lstm_3',
                   units = 64,
                   return_sequences = True))
    
    model.add(Dropout(0.1, name = 'dropout_3'))
    model.add(BatchNormalization(name = 'batch_norm_3'))
    
    model.add(LSTM(name ='lstm_4',
                   units = sta_num,
                   return_sequences = True))
    
    model.add(TimeDistributed(Dense(units=sta_num, name = 'dense_1', activation = 'linear')))
    model.add(Flatten())
    
    #compile model
    model.compile(optimizer=config.optimizer,metrics=config.metrics,loss=config.loss)
    if summary:
        model.summary()
    
    return model

def purelstm(config:ConfigStruct,summary=True)->Model:
    #import what you need
    from keras.layers import BatchNormalization,LSTM,Dropout,RepeatVector,TimeDistributed,Dense,Flatten

    #get settings from config struct
    input_timestep=config.input_timestep
    sta_num=config.sta_num
    output_timestep=config.output_timestep

    #start build model
    model=models.Sequential(name=config.model_name)
    model.add(BatchNormalization(name = 'batch_norm_0', input_shape = (input_timestep,sta_num)))
    model.add(LSTM(name ='lstm_1',
                   units = 64,
                   return_sequences = True))
    
    model.add(Dropout(0.2, name = 'dropout_1'))
    model.add(BatchNormalization(name = 'batch_norm_1'))

    model.add(LSTM(name ='lstm_2',
                   units = 64,
                   return_sequences = False))
    
    model.add(Dropout(0.1, name = 'dropout_2'))
    model.add(BatchNormalization(name = 'batch_norm_2'))
    
    model.add(RepeatVector(output_timestep))
    
    model.add(LSTM(name ='lstm_3',
                   units = 64,
                   return_sequences = True))
    
    model.add(Dropout(0.1, name = 'dropout_3'))
    model.add(BatchNormalization(name = 'batch_norm_3'))
    
    model.add(LSTM(name ='lstm_4',
                   units = sta_num,
                   return_sequences = True))
    
    model.add(TimeDistributed(Dense(units=sta_num, name = 'dense_1', activation = 'linear')))
    model.add(Flatten())
    
    #compile model
    model.compile(optimizer=config.optimizer,metrics=config.metrics,loss=config.loss)
    if summary:
        model.summary()
    
    return model

def puredense(config:ConfigStruct,summary=True)->Model:
    #import what you need
    from keras.models import Sequential
    from keras.layers import BatchNormalization,LSTM,Dropout,RepeatVector,TimeDistributed,Dense,Flatten

    #get settings from config struct
    input_timestep=config.input_timestep
    sta_num=config.sta_num
    output_timestep=config.output_timestep

    #start build model
    model=Sequential(name=config.model_name)
    model.add(BatchNormalization(input_shape=(input_timestep,sta_num)))
    model.add(TimeDistributed(Dense(units=64)))
    model.add(BatchNormalization())
    model.add(TimeDistributed(Dense(units=64)))
    model.add(Flatten())
    model.add(RepeatVector(output_timestep))
    model.add(TimeDistributed(Dense(64)))
    model.add(BatchNormalization())
    model.add(TimeDistributed(Dense(sta_num,activation='linear')))
    model.add(Flatten())
    #compile model
    model.compile(optimizer=config.optimizer,metrics=config.metrics,loss=config.loss)
    if summary:
        model.summary()
    
    return model

def convlstm(config:ConfigStruct,summary=True)->Model:
    #import what you need
    from keras.models import Sequential
    from keras.layers import BatchNormalization,LSTM,Dropout,RepeatVector,TimeDistributed,Dense,Flatten,Reshape,ConvLSTM2D

    #get settings from config struct
    input_timestep=config.input_timestep
    sta_num=config.sta_num
    output_timestep=config.output_timestep

    #start build model
    model=Sequential(name=config.model_name)
    model.add(BatchNormalization(input_shape=(input_timestep,sta_num)))
    model.add(Reshape((input_timestep,sta_num,1,1)))
    model.add(ConvLSTM2D(filters=64,kernel_size=(10,1),padding='same',return_sequences=True))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())
    model.add(ConvLSTM2D(filters=64,kernel_size=(5,1),padding='same',return_sequences=False))
    model.add(Dropout(0.1))
    model.add(BatchNormalization())
    model.add(Flatten())
    model.add(RepeatVector(output_timestep))
    model.add(Reshape((output_timestep,sta_num,1,64)))
    model.add(ConvLSTM2D(filters=64,kernel_size=(10,1),padding='same',return_sequences=True))
    model.add(Dropout(0.1))
    model.add(BatchNormalization())
    model.add(ConvLSTM2D(filters=64,kernel_size=(5,1),padding='same',return_sequences=True))
    model.add(TimeDistributed(Dense(1,activation='relu')))
    model.add(Flatten())
    #compile model
    model.compile(optimizer=config.optimizer,metrics=config.metrics,loss=config.loss)
    if summary:
        model.summary()
    
    return model