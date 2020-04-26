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

def stdn()->Model:

    class attention_layer(layers.Layer):#simplest attention layer
        def __init__(self, **kwargs):
            super(attention_layer, self).__init__(**kwargs)

        def build(self,input_shape):#input_shape=(,9,64)
            assert len(input_shape)==3
            self.W=self.add_weight(name='attr_weight',shape=(input_shape[1],input_shape[2]),initializer='uniform',trainable=True)
            self.b=self.add_weight(name='attr_bias',shape=(input_shape[2],),initializer='uniform',trainable=True)
            super(attention_layer,self).build(input_shape)

        def call(self,inputs):
            x=K.permute_dimensions(inputs,(0,2,1))#(,9,64)->(,64,9)
            a=K.dot(x,self.W) #(,64,9).(,9,64)->(64,64)
            a_prob=K.bias_add(a,self.b)#(64,64)
            a=K.tanh(a_prob)#(64,64)
            a=K.softmax(a,axis=1)#(64,64)
            a=a*a_prob
            a=K.permute_dimensions(a,(0,2,1))
            a=K.sum(a,axis=1)
            return a

        def compute_output_shape(self, input_shape):
            return input_shape[0], input_shape[2]
    
    lt=layers.Input(shape=(3,9,20))
    st=layers.Input(shape=(8,20))
    wd=layers.Input(shape=(4,1))#weekday
    ef=layers.Input(shape=(4,20))#external features like weather

    #first use conv to obtain spacial dependency
    y1=layers.Reshape((27,20,1),name='lt_reshape_1')(lt)#I don't want to use POI ,it increase largely on data ,
    y1=layers.TimeDistributed(layers.Conv1D(filters=64,kernel_size=6,padding='same'))(y1)
    y1=layers.Activation('relu')(y1)
    y1=layers.Reshape((3,9,20*64),name='lt_reshape_2')(y1)
    y1=layers.TimeDistributed(layers.LSTM(64,return_sequences=True,dropout=0.1,recurrent_dropout=0.1))(y1)
    y1=layers.TimeDistributed(layers.LSTM(64,return_sequences=True,dropout=0.1,recurrent_dropout=0.1))(y1)
    y1=layers.TimeDistributed(attention_layer())(y1)

    y2=layers.Reshape((8,20,1))(st)
    y2=layers.TimeDistributed(layers.Conv1D(filters=64,kernel_size=6,padding='same'))(y2)
    y2=layers.Activation('relu')(y2)
    y2=layers.Reshape((8,20*64))(y2)
    y2=layers.LSTM(64,return_sequences=True,dropout=0.1,recurrent_dropout=0.1)(y2)
    y2=layers.LSTM(64,return_sequences=True,dropout=0.1,recurrent_dropout=0.1)(y2)
    y2=attention_layer()(y2)
    y2=layers.Reshape((1,64))(y2)

    y3=layers.TimeDistributed(layers.Dense(1,activation='relu'))(ef)

    y=layers.concatenate([y1,y2],axis=1)
    y=layers.concatenate([y,wd,y3])
    y=layers.LSTM(64,return_sequences=True,dropout=0.1,recurrent_dropout=0.1)(y)
    y=layers.LSTM(64,return_sequences=True,dropout=0.1,recurrent_dropout=0.1)(y)
    y=attention_layer()(y)
    y=layers.Dense(40)(y)
    
    model=Model(inputs=[lt,st,wd,ef],outputs=[y],name='stdn')
    model.compile(optimizer='adam',loss='mse')
    #model.summary()
    return model

if __name__ == "__main__":
    from keras.utils import plot_model
    model=stdn()
    plot_model(model,to_file='stdn_model.png',show_shapes=True)
