import numpy as np
import pandas as pd
from matplotlib import pyplot
import datetime
import os
import sys

np.random.seed(seed=5)

def stdn_like_model_test():
    from keras import layers
    from keras import Model
    from keras import models
    from keras import backend as K
    from keras.callbacks import EarlyStopping,ReduceLROnPlateau
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.metrics import mean_squared_error,mean_absolute_error

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

    def build_model():
        lt=layers.Input(shape=(27,20))
        st=layers.Input(shape=(8,20))
        wd=layers.Input(shape=(4,1))#weekday
        #ef=layers.Input(shape=(4,24))#external features like weather

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

        #y3=layers.TimeDistributed(layers.Dense(1,activation='relu'))(ef)

        y=layers.concatenate([y1,y2],axis=1)
        y=layers.concatenate([y,wd])
        y=layers.Flatten()(y)
        y=layers.Dense(20)(y)#only one output for now

        model=Model(inputs=[lt,st,wd],outputs=[y],name='stdn')
        model.compile(optimizer='adam',loss='mse')
        return model
    
    def generate_data():
        def onehot_external_features()->(dict,dict):
            ef=pd.read_csv('weather.csv',parse_dates=['date'])
            enc=OneHotEncoder()
            weather_list=list()
            wd_list=list()#wind direction
            for row in ef.iterrows():
                if row[1]['dw'] not in weather_list:
                    weather_list.append(row[1]['dw'])
                if row[1]['nw'] not in weather_list:
                    weather_list.append(row[1]['nw'])
                if row[1]['wd'] not in wd_list:
                    wd_list.append(row[1]['wd'])
            weather_array=np.array(weather_list).reshape((-1,1))
            wd_array=np.array(wd_list).reshape((-1,1))
            weather_array=enc.fit_transform(weather_array).toarray()
            wd_array=enc.fit_transform(wd_array).toarray()

            weather_dict=dict()
            for i in range(0,len(weather_array)):
                weather_dict[weather_list[i]]=list(weather_array[i])
            wd_dict=dict()
            for i in range(0,len(wd_array)):
                wd_dict[wd_list[i]]=list(wd_array[i])
            return weather_dict,wd_dict
        print('generate data')
        #raw=pd.read_csv('tt_dataset_mymodel.csv',parse_dates=['start_time'])
        raw=pd.read_csv('pf_dataset_mymodel.csv',parse_dates=['start_time'])
        index=raw.pop('start_time')
        len_raw=int(len(raw)/(3*9+8+2))
        splitpoint=int(len_raw*0.7)
        raw_numpy:np.ndarray=raw.to_numpy().reshape((
            len_raw,(3*9+8+2),20
        ))
        train=raw_numpy[:splitpoint].flatten().reshape((-1,1))
        test=raw_numpy[splitpoint:].flatten().reshape((-1,1))
        
        mms=MinMaxScaler()
        
        train=mms.fit_transform(train)
        train=train.reshape((splitpoint,(3*9+8+2),20))
        lt_train=train[:,:3*9]
        st_train=train[:,3*9:3*9+8]
        y_train=train[:,-2:]

        test=mms.transform(test)
        test=test.reshape((len_raw-splitpoint,(3*9+8+2),20))
        lt_test=test[:,:3*9]
        st_test=test[:,3*9:3*9+8]
        y_test=test[:,-2:]

        print('generate external data')
        ef=pd.read_csv('weather.csv',parse_dates=['date'])
        ef.set_index(['date'],inplace=True)
        weather_dict,wd_dict=onehot_external_features()
        weekday=list()
        ef_list=list()
        for i in range(0,len(index),3*9+8+2):
            for j in range(0,3*9+8,9):
                ts:pd.Timestamp=index[i+j]
                if ts.weekday()>=5:
                    weekday.append(1)
                else:
                    weekday.append(0)
                #weekday.append(ts.weekday())
                ef_list.append(ef.at[ts.date(),'AQI'])
                ef_list+=weather_dict[ef.at[ts.date(),'dw']]
                ef_list+=weather_dict[ef.at[ts.date(),'nw']]
                ef_list.append(ef.at[ts.date(),'ht'])
                ef_list.append(ef.at[ts.date(),'lt'])
                ef_list+=wd_dict[ef.at[ts.date(),'wd']]
                ef_list.append(ef.at[ts.date(),'wf'])
            print('\r\tdone ',i,'/',len(index),sep='',end='')
        print('')
        weekday=np.array(weekday).reshape((
            len_raw,4,1
        ))
        ef=np.array(ef_list).reshape((
            len_raw,4,int(len(ef_list)/(len_raw*4))
        ))
        #print(type(index))#<class 'pandas.core.series.Series'>
        #print(point_num)
        #print(weekday[0])
        #print(ef[0])
        data={
            'lt_train':lt_train,
            'st_train':st_train,
            'weekday_train':weekday[:splitpoint],
            'ef_train':ef[:splitpoint],
            'y_train':y_train,
            'lt_test':lt_test,
            'st_test':st_test,
            'weekday_test':weekday[splitpoint:],
            'ef_test':ef[splitpoint:],
            'y_test':y_test,
            'mms':mms
        }
        return data

    model=build_model()
    model.summary()
    data=generate_data()
    x_train=[data['lt_train'],data['st_train'],data['weekday_train']]
    y_train=[data['y_train'][:,0]]#for now only predict one step
    x_test=[data['lt_test'],data['st_test'],data['weekday_test']]
    y_test=[data['y_test'][:,0]]
    earlystop=EarlyStopping(patience=20,restore_best_weights=True)
    reduce_lr=ReduceLROnPlateau(patience=5)
    callback_list=[earlystop,reduce_lr]
    model.fit(x_train,y_train,batch_size=16,epochs=2000,validation_data=[x_test,y_test],callbacks=callback_list)
    model.save_weights('stdn_weight.h5')
    mms:MinMaxScaler=data['mms']
    y_pred=model.predict(x_test)
    y_true=mms.inverse_transform(y_test[0].flatten().reshape(-1,1)).flatten()
    y_pred=mms.inverse_transform(y_pred.flatten().reshape(-1,1)).flatten()
    print('mse:',mean_squared_error(y_true,y_pred))
    print('mae:',mean_absolute_error(y_true,y_pred))

def convlstm2d_model_test():
    from keras import layers
    from keras import Model
    from keras import models
    from keras import backend as K
    from keras.callbacks import EarlyStopping,ReduceLROnPlateau
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.metrics import mean_squared_error,mean_absolute_error

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

    def build_model():
        '''
        #lt=layers.Input(shape=(27,20))
        st=layers.Input(shape=(8,20))
        #wd=layers.Input(shape=(4,1))#weekday
        
        y=layers.Reshape((8,20,1,1))(st)
        y=layers.BatchNormalization()(y)
        y=layers.ConvLSTM2D(filters=64,kernel_size=(10,1),padding='same',return_sequences=True)(y)
        y=layers.Dropout(0.2)(y)
        y=layers.BatchNormalization()(y)
        y=layers.ConvLSTM2D(filters=64,kernel_size=(5,1),padding='same',return_sequences=False)(y)
        y=layers.Dropout(0.1)(y)
        y=layers.BatchNormalization()(y)
        y=layers.Flatten()(y)
        y=layers.RepeatVector(1)(y)
        y=layers.Reshape((1,20,1,64))(y)
        y=layers.ConvLSTM2D(filters=64,kernel_size=(10,1),padding='same',return_sequences=True)(y)
        y=layers.Dropout(0.1)(y)
        y=layers.BatchNormalization()(y)
        y=layers.ConvLSTM2D(filters=64,kernel_size=(5,1),padding='same',return_sequences=True)(y)
        y=layers.TimeDistributed(layers.Dense(1,activation='relu'))(y)
        y=layers.Flatten()(y)
        
        model=Model(inputs=[st],outputs=[y],name='stdn')
        '''
        from keras.models import Sequential
        from keras.layers import BatchNormalization,Reshape,ConvLSTM2D,Dropout,Flatten,TimeDistributed,Dense,RepeatVector
        input_timestep=8
        sta_num=20
        output_timestep=1
        model=Sequential(name='convlstm2d')
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
        model.compile(optimizer='adam',loss='mse')
        return model
    
    def generate_data():
        def onehot_external_features()->(dict,dict):
            ef=pd.read_csv('weather.csv',parse_dates=['date'])
            enc=OneHotEncoder()
            weather_list=list()
            wd_list=list()#wind direction
            for row in ef.iterrows():
                if row[1]['dw'] not in weather_list:
                    weather_list.append(row[1]['dw'])
                if row[1]['nw'] not in weather_list:
                    weather_list.append(row[1]['nw'])
                if row[1]['wd'] not in wd_list:
                    wd_list.append(row[1]['wd'])
            weather_array=np.array(weather_list).reshape((-1,1))
            wd_array=np.array(wd_list).reshape((-1,1))
            weather_array=enc.fit_transform(weather_array).toarray()
            wd_array=enc.fit_transform(wd_array).toarray()

            weather_dict=dict()
            for i in range(0,len(weather_array)):
                weather_dict[weather_list[i]]=list(weather_array[i])
            wd_dict=dict()
            for i in range(0,len(wd_array)):
                wd_dict[wd_list[i]]=list(wd_array[i])
            return weather_dict,wd_dict
        print('generate data')
        raw=pd.read_csv('tt_dataset_mymodel.csv',parse_dates=['start_time'])
        #raw=pd.read_csv('pf_dataset_mymodel.csv',parse_dates=['start_time'])
        index=raw.pop('start_time')
        len_raw=int(len(raw)/(3*9+8+2))
        splitpoint=int(len_raw*0.7)
        raw_numpy:np.ndarray=raw.to_numpy().reshape((
            len_raw,(3*9+8+2),20
        ))
        train=raw_numpy[:splitpoint].flatten().reshape((-1,1))
        test=raw_numpy[splitpoint:].flatten().reshape((-1,1))
        
        mms=MinMaxScaler()
        
        train=mms.fit_transform(train)
        train=train.reshape((splitpoint,(3*9+8+2),20))
        lt_train=train[:,:3*9]
        st_train=train[:,3*9:3*9+8]
        y_train=train[:,-2:]

        test=mms.transform(test)
        test=test.reshape((len_raw-splitpoint,(3*9+8+2),20))
        lt_test=test[:,:3*9]
        st_test=test[:,3*9:3*9+8]
        y_test=test[:,-2:]

        print('generate external data')
        ef=pd.read_csv('weather.csv',parse_dates=['date'])
        ef.set_index(['date'],inplace=True)
        weather_dict,wd_dict=onehot_external_features()
        weekday=list()
        ef_list=list()
        for i in range(0,len(index),3*9+8+2):
            for j in range(0,3*9+8,9):
                ts:pd.Timestamp=index[i+j]
                if ts.weekday()>=5:
                    weekday.append(1)
                else:
                    weekday.append(0)
                #weekday.append(ts.weekday())
                ef_list.append(ef.at[ts.date(),'AQI'])
                ef_list+=weather_dict[ef.at[ts.date(),'dw']]
                ef_list+=weather_dict[ef.at[ts.date(),'nw']]
                ef_list.append(ef.at[ts.date(),'ht'])
                ef_list.append(ef.at[ts.date(),'lt'])
                ef_list+=wd_dict[ef.at[ts.date(),'wd']]
                ef_list.append(ef.at[ts.date(),'wf'])
            print('\r\tdone ',i,'/',len(index),sep='',end='')
        print('')
        weekday=np.array(weekday).reshape((
            len_raw,4,1
        ))
        ef=np.array(ef_list).reshape((
            len_raw,4,int(len(ef_list)/(len_raw*4))
        ))
        #print(type(index))#<class 'pandas.core.series.Series'>
        #print(point_num)
        #print(weekday[0])
        #print(ef[0])
        data={
            'lt_train':lt_train,
            'st_train':st_train,
            'weekday_train':weekday[:splitpoint],
            'ef_train':ef[:splitpoint],
            'y_train':y_train,
            'lt_test':lt_test,
            'st_test':st_test,
            'weekday_test':weekday[splitpoint:],
            'ef_test':ef[splitpoint:],
            'y_test':y_test,
            'mms':mms
        }
        return data

    model=build_model()
    #model.summary()
    data=generate_data()
    x_train=[data['st_train']]
    y_train=[data['y_train'][:,0]]#for now only predict one step
    x_test=[data['st_test']]
    y_test=[data['y_test'][:,0]]
    earlystop=EarlyStopping(patience=20,restore_best_weights=True)
    reduce_lr=ReduceLROnPlateau(patience=5)
    callback_list=[earlystop,reduce_lr]
    metrics=model.fit(x_train,y_train,batch_size=16,epochs=2000,validation_data=[x_test,y_test],callbacks=callback_list)
    model.save_weights('convlstm2d_weight.h5')

    pyplot.Figure(dpi=92)
    pyplot.title('history metrics')
    pyplot.xlabel('epoch')
    pyplot.plot(metrics.history['loss'],label='loss',color='r',linestyle=':',linewidth=1.0)
    pyplot.plot(metrics.history['val_loss'],label='val_loss',color='r',linestyle='-',linewidth=1.0)
    pyplot.legend()
    pyplot.show()
    pyplot.savefig('loss.png')

    mms:MinMaxScaler=data['mms']
    y_pred=model.predict(x_test)
    y_true=mms.inverse_transform(y_test[0].flatten().reshape(-1,1)).flatten()
    y_pred=mms.inverse_transform(y_pred.flatten().reshape(-1,1)).flatten()
    print('mse:',mean_squared_error(y_true,y_pred))
    print('mae:',mean_absolute_error(y_true,y_pred))

def cnn_lstm_model_test():
    from keras import layers
    from keras import Model
    from keras import models
    from keras import backend as K
    from keras.callbacks import EarlyStopping,ReduceLROnPlateau
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.metrics import mean_squared_error,mean_absolute_error

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

    def build_model():
        lt=layers.Input(shape=(27,20))
        st=layers.Input(shape=(8,20))
        wd=layers.Input(shape=(4,1))#weekday
        
        y1=layers.BatchNormalization()(lt)
        y1=layers.Reshape((27,20,1))(y1)
        y1=layers.TimeDistributed(layers.Conv1D(filters=64,kernel_size=10,padding='same'))(y1)
        y1=layers.Reshape((3,9,20*64))(y1)
        y1=layers.TimeDistributed(layers.LSTM(64,return_sequences=True))(y1)
        y1=layers.Dropout(0.2)(y1)
        y1=layers.BatchNormalization()(y1)
        y1=layers.Reshape((27,64,1))(y1)
        y1=layers.TimeDistributed(layers.Conv1D(filters=64,kernel_size=10,padding='same'))(y1)
        y1=layers.Reshape((3,9,64*64))(y1)
        y1=layers.TimeDistributed(layers.LSTM(64,return_sequences=True))(y1)
        y1=layers.Dropout(0.1)(y1)
        y1=layers.TimeDistributed(attention_layer())(y1)#->(3,1,64)

        y2=layers.BatchNormalization()(st)
        y2=layers.Reshape((8,20,1))(y2)
        y2=layers.TimeDistributed(layers.Conv1D(filters=64,kernel_size=10,padding='same'))(y2)
        y2=layers.Reshape((8,20*64))(y2)
        y2=layers.LSTM(64,return_sequences=True)(y2)
        y2=layers.Dropout(0.2)(y2)
        y2=layers.BatchNormalization()(y2)
        y2=layers.Reshape((8,64,1))(y2)
        y2=layers.TimeDistributed(layers.Conv1D(filters=64,kernel_size=5,padding='same'))(y2)
        y2=layers.Reshape((8,64*64))(y2)
        y2=layers.LSTM(64,return_sequences=True)(y2)#->(8,64)
        y2=layers.Dropout(0.1)(y2)
        y2=attention_layer()(y2)#->(1,64)
        y2=layers.Reshape((1,64))(y2)
        
        y=layers.concatenate([y1,y2],axis=1)
        y=layers.BatchNormalization()(y)
        y=layers.concatenate([y,wd])
        y=layers.Flatten()(y)
        y=layers.Dense(20,activation='relu')(y)
        
        model=Model(inputs=[lt,st,wd],outputs=[y],name='stdn')
        model.compile(optimizer='adam',loss='mse')
        return model
    
    def generate_data():
        def onehot_external_features()->(dict,dict):
            ef=pd.read_csv('weather.csv',parse_dates=['date'])
            enc=OneHotEncoder()
            weather_list=list()
            wd_list=list()#wind direction
            for row in ef.iterrows():
                if row[1]['dw'] not in weather_list:
                    weather_list.append(row[1]['dw'])
                if row[1]['nw'] not in weather_list:
                    weather_list.append(row[1]['nw'])
                if row[1]['wd'] not in wd_list:
                    wd_list.append(row[1]['wd'])
            weather_array=np.array(weather_list).reshape((-1,1))
            wd_array=np.array(wd_list).reshape((-1,1))
            weather_array=enc.fit_transform(weather_array).toarray()
            wd_array=enc.fit_transform(wd_array).toarray()

            weather_dict=dict()
            for i in range(0,len(weather_array)):
                weather_dict[weather_list[i]]=list(weather_array[i])
            wd_dict=dict()
            for i in range(0,len(wd_array)):
                wd_dict[wd_list[i]]=list(wd_array[i])
            return weather_dict,wd_dict
        print('generate data')
        raw=pd.read_csv('tt_dataset_mymodel.csv',parse_dates=['start_time'])
        #raw=pd.read_csv('pf_dataset_mymodel.csv',parse_dates=['start_time'])
        index=raw.pop('start_time')
        len_raw=int(len(raw)/(3*9+8+2))
        splitpoint=int(len_raw*0.7)
        raw_numpy:np.ndarray=raw.to_numpy().reshape((
            len_raw,(3*9+8+2),20
        ))
        train=raw_numpy[:splitpoint].flatten().reshape((-1,1))
        test=raw_numpy[splitpoint:].flatten().reshape((-1,1))
        
        mms=MinMaxScaler()
        
        train=mms.fit_transform(train)
        train=train.reshape((splitpoint,(3*9+8+2),20))
        lt_train=train[:,:3*9]
        st_train=train[:,3*9:3*9+8]
        y_train=train[:,-2:]

        test=mms.transform(test)
        test=test.reshape((len_raw-splitpoint,(3*9+8+2),20))
        lt_test=test[:,:3*9]
        st_test=test[:,3*9:3*9+8]
        y_test=test[:,-2:]

        print('generate external data')
        ef=pd.read_csv('weather.csv',parse_dates=['date'])
        ef.set_index(['date'],inplace=True)
        weather_dict,wd_dict=onehot_external_features()
        weekday=list()
        ef_list=list()
        for i in range(0,len(index),3*9+8+2):
            for j in range(0,3*9+8,9):
                ts:pd.Timestamp=index[i+j]
                if ts.weekday()>=5:
                    weekday.append(1)
                else:
                    weekday.append(0)
                #weekday.append(ts.weekday())
                ef_list.append(ef.at[ts.date(),'AQI'])
                ef_list+=weather_dict[ef.at[ts.date(),'dw']]
                ef_list+=weather_dict[ef.at[ts.date(),'nw']]
                ef_list.append(ef.at[ts.date(),'ht'])
                ef_list.append(ef.at[ts.date(),'lt'])
                ef_list+=wd_dict[ef.at[ts.date(),'wd']]
                ef_list.append(ef.at[ts.date(),'wf'])
            print('\r\tdone ',i,'/',len(index),sep='',end='')
        print('')
        weekday=np.array(weekday).reshape((
            len_raw,4,1
        ))
        ef=np.array(ef_list).reshape((
            len_raw,4,int(len(ef_list)/(len_raw*4))
        ))
        #print(type(index))#<class 'pandas.core.series.Series'>
        #print(point_num)
        #print(weekday[0])
        #print(ef[0])
        data={
            'lt_train':lt_train,
            'st_train':st_train,
            'weekday_train':weekday[:splitpoint],
            'ef_train':ef[:splitpoint],
            'y_train':y_train,
            'lt_test':lt_test,
            'st_test':st_test,
            'weekday_test':weekday[splitpoint:],
            'ef_test':ef[splitpoint:],
            'y_test':y_test,
            'mms':mms
        }
        return data

    model=build_model()
    model.summary()
    from keras.utils import plot_model
    plot_model(model,show_shapes=True)
    
    data=generate_data()
    x_train=[data['st_train']]
    y_train=[data['y_train'][:,0]]#for now only predict one step
    x_test=[data['st_test']]
    y_test=[data['y_test'][:,0]]
    earlystop=EarlyStopping(patience=20,restore_best_weights=True)
    reduce_lr=ReduceLROnPlateau(patience=5)
    callback_list=[earlystop,reduce_lr]
    metrics=model.fit(x_train,y_train,batch_size=16,epochs=2000,validation_data=[x_test,y_test],callbacks=callback_list)
    model.save_weights('cnn_lstm_weight.h5')

    pyplot.Figure(dpi=92)
    pyplot.title('history metrics')
    pyplot.xlabel('epoch')
    pyplot.plot(metrics.history['loss'],label='loss',color='r',linestyle=':',linewidth=1.0)
    pyplot.plot(metrics.history['val_loss'],label='val_loss',color='r',linestyle='-',linewidth=1.0)
    pyplot.legend()
    pyplot.show()
    pyplot.savefig('loss.png')

    mms:MinMaxScaler=data['mms']
    y_pred=model.predict(x_test)
    y_true=mms.inverse_transform(y_test[0].flatten().reshape(-1,1)).flatten()
    y_pred=mms.inverse_transform(y_pred.flatten().reshape(-1,1)).flatten()
    print('mse:',mean_squared_error(y_true,y_pred))
    print('mae:',mean_absolute_error(y_true,y_pred))

def test_model_test(datasource='tt_dataset_mymodel.csv'):
    from keras import layers
    from keras import Model
    from keras import models
    from keras import backend as K
    from keras.callbacks import EarlyStopping,ReduceLROnPlateau
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.metrics import mean_squared_error,mean_absolute_error

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

    def build_model():
        lt=layers.Input(shape=(27,20))
        st=layers.Input(shape=(8,20))
        wd=layers.Input(shape=(4,1))#weekday
        
        #first use conv to obtain spacial dependency
        y1=layers.Reshape((3,9,20,1,1),name='lt_reshape_1')(lt)#I don't want to use POI ,it increase largely on data ,
        y1=layers.TimeDistributed(layers.ConvLSTM2D(filters=64,kernel_size=(10,1),padding='same',return_sequences=True))(y1)
        y1=layers.Activation('relu')(y1)
        y1=layers.TimeDistributed(layers.ConvLSTM2D(filters=64,kernel_size=(5,1),padding='same',return_sequences=True))(y1)
        y1=layers.Reshape((3,9,20*64))(y1)
        y1=layers.Dense(20,activation='relu')(y1)
        y1=layers.TimeDistributed(attention_layer())(y1)

        y2=layers.Reshape((8,20,1,1))(st)
        y2=layers.ConvLSTM2D(filters=64,kernel_size=(10,1),padding='same',return_sequences=True)(y2)
        y2=layers.Activation('relu')(y2)
        y2=layers.ConvLSTM2D(filters=64,kernel_size=(5,1),padding='same',return_sequences=True)(y2)
        y2=layers.Reshape((1,8,20*64))(y2)
        y2=layers.Dense(20,activation='relu')(y2)
        y2=layers.TimeDistributed(attention_layer())(y2)

        y=layers.concatenate([y1,y2],axis=1)
        y=layers.concatenate([y,wd])
        y=layers.Flatten()(y)
        y=layers.Dense(20)(y)#only one output for now
        
        model=Model(inputs=[lt,st,wd],outputs=[y],name='stdn')
        model.compile(optimizer='adam',loss='mse')
        return model
    
    def generate_data(datasource:str):
        def onehot_external_features()->(dict,dict):
            ef=pd.read_csv('weather.csv',parse_dates=['date'])
            enc=OneHotEncoder()
            weather_list=list()
            wd_list=list()#wind direction
            for row in ef.iterrows():
                if row[1]['dw'] not in weather_list:
                    weather_list.append(row[1]['dw'])
                if row[1]['nw'] not in weather_list:
                    weather_list.append(row[1]['nw'])
                if row[1]['wd'] not in wd_list:
                    wd_list.append(row[1]['wd'])
            weather_array=np.array(weather_list).reshape((-1,1))
            wd_array=np.array(wd_list).reshape((-1,1))
            weather_array=enc.fit_transform(weather_array).toarray()
            wd_array=enc.fit_transform(wd_array).toarray()

            weather_dict=dict()
            for i in range(0,len(weather_array)):
                weather_dict[weather_list[i]]=list(weather_array[i])
            wd_dict=dict()
            for i in range(0,len(wd_array)):
                wd_dict[wd_list[i]]=list(wd_array[i])
            return weather_dict,wd_dict
        print('generate data')
        raw=pd.read_csv(datasource,parse_dates=['start_time'])
        #raw=pd.read_csv('pf_dataset_mymodel.csv',parse_dates=['start_time'])
        index=raw.pop('start_time')
        len_raw=int(len(raw)/(3*9+8+2))
        splitpoint=int(len_raw*0.7)
        raw_numpy:np.ndarray=raw.to_numpy().reshape((
            len_raw,(3*9+8+2),20
        ))
        train=raw_numpy[:splitpoint].flatten().reshape((-1,1))
        test=raw_numpy[splitpoint:].flatten().reshape((-1,1))
        
        mms=MinMaxScaler()
        
        train=mms.fit_transform(train)
        train=train.reshape((splitpoint,(3*9+8+2),20))
        lt_train=train[:,:3*9]
        st_train=train[:,3*9:3*9+8]
        y_train=train[:,-2:]

        test=mms.transform(test)
        test=test.reshape((len_raw-splitpoint,(3*9+8+2),20))
        lt_test=test[:,:3*9]
        st_test=test[:,3*9:3*9+8]
        y_test=test[:,-2:]

        print('generate external data')
        ef=pd.read_csv('weather.csv',parse_dates=['date'])
        ef.set_index(['date'],inplace=True)
        weather_dict,wd_dict=onehot_external_features()
        weekday=list()
        ef_list=list()
        for i in range(0,len(index),3*9+8+2):
            for j in range(0,3*9+8,9):
                ts:pd.Timestamp=index[i+j]
                if ts.weekday()>=5:
                    weekday.append(1)
                else:
                    weekday.append(0)
                #weekday.append(ts.weekday())
                ef_list.append(ef.at[ts.date(),'AQI'])
                ef_list+=weather_dict[ef.at[ts.date(),'dw']]
                ef_list+=weather_dict[ef.at[ts.date(),'nw']]
                ef_list.append(ef.at[ts.date(),'ht'])
                ef_list.append(ef.at[ts.date(),'lt'])
                ef_list+=wd_dict[ef.at[ts.date(),'wd']]
                ef_list.append(ef.at[ts.date(),'wf'])
            print('\r\tdone ',i,'/',len(index),sep='',end='')
        print('')
        weekday=np.array(weekday).reshape((
            len_raw,4,1
        ))
        ef=np.array(ef_list).reshape((
            len_raw,4,int(len(ef_list)/(len_raw*4))
        ))
        #print(type(index))#<class 'pandas.core.series.Series'>
        #print(point_num)
        #print(weekday[0])
        #print(ef[0])
        data={
            'lt_train':lt_train,
            'st_train':st_train,
            'weekday_train':weekday[:splitpoint],
            'ef_train':ef[:splitpoint],
            'y_train':y_train,
            'lt_test':lt_test,
            'st_test':st_test,
            'weekday_test':weekday[splitpoint:],
            'ef_test':ef[splitpoint:],
            'y_test':y_test,
            'mms':mms
        }
        return data

    model=build_model()
    model.summary()
    from keras.utils import plot_model
    plot_model(model,show_shapes=True)
    '''
    data=generate_data(datasource)
    x_train=[data['st_train']]
    y_train=[data['y_train'][:,0]]#for now only predict one step
    x_test=[data['st_test']]
    y_test=[data['y_test'][:,0]]
    earlystop=EarlyStopping(patience=20,restore_best_weights=True)
    reduce_lr=ReduceLROnPlateau(patience=5)
    callback_list=[earlystop,reduce_lr]
    metrics=model.fit(x_train,y_train,batch_size=16,epochs=2000,validation_data=[x_test,y_test],callbacks=callback_list)
    model.save_weights('cnn_lstm_weight.h5')

    pyplot.Figure(dpi=92)
    pyplot.title('history metrics')
    pyplot.xlabel('epoch')
    pyplot.plot(metrics.history['loss'],label='loss',color='r',linestyle=':',linewidth=1.0)
    pyplot.plot(metrics.history['val_loss'],label='val_loss',color='r',linestyle='-',linewidth=1.0)
    pyplot.legend()
    pyplot.show()
    pyplot.savefig('loss.png')

    mms:MinMaxScaler=data['mms']
    y_pred=model.predict(x_test)
    y_true=mms.inverse_transform(y_test[0].flatten().reshape(-1,1)).flatten()
    y_pred=mms.inverse_transform(y_pred.flatten().reshape(-1,1)).flatten()
    print('mse:',mean_squared_error(y_true,y_pred))
    print('mae:',mean_absolute_error(y_true,y_pred))
    '''


if __name__ == "__main__":
    print('comparison.py')
    print('！！！一定要确认生成的数据是tt还是pf数据，修改权重文件路径！！！')
    #stdn_like_model_test()
    #if with ef,weekday and lstmlayers ->base with ef weekday and ll
    #mse: 1389.2044148417294#mae: 25.699903068126847
    #if no ef and weekday and no additional lstm layers ->base
    #mse: 1004.4012645637565#mae: 21.74309913973737
    #if no ef and no additional lstm layers ->base with weekday
    #mse: 1158.7831961560807#mae: 23.29928049199119
    #if no ef ->base with weekday ans lstmlayers ->lstmlayers is bad thing
    #mse: 1390.2886733245339#mae: 25.64868130994199
    #if no weekday no ll -> base with ef->very bad...
    #mse: 7638.241481057416#mae: 42.351122696165

    #basically saying that we don't need ef,,for now it's just a bad thing
    #if just with simplified weekday
    #mse: 995.9403607166892#mae: 21.64213124298836
    #so from here we know that ,we simply don't have enough data
    # we need tons of data for lstm to remember all kinds of situation

    #if we use the weekday data as a part of encoder result->we already using

    #and use the best to test on pf data ,is is still good?,convlstm is about 2.1 person 
    #mse: 9.44085464065043#mae: 2.0973220654015443 close to convlstm but faster much faster/xk

    #convlstm2d_model_test()
    #same with the one in the paper but with 8 timestep instead of 32
    #short term dependency only
    #for pf data test 
    #mse: 9.743627215739645#mae: 2.080584265229
    #for tt data test ->the result was too bad... not knowing the reason
    #mse: 18871.005344470148#mae: 126.19269104477613 because the dataset is better for those who has both long term and short term dependency
    #if it goes with 32 timesteps and not filling n/a data with 0 , and transcode seconds to minutes
    #the mse shall be about 0.2558 and mae is 0.3521 -> about 21.126s better than stdn model above

    #cnn_lstm_model_test()
    #basically already knows that is going to be quite bad ,but just test in case there something still not know
    #pf data
    #mse: 29.313801778764194#mae: 3.601255823062427
    #tt data,not knowing why, but just don't work
    #anyway the basic model that looks like "convlstm1d" does not work well
    #partially because test data was generated based on long short term 
    #after changing it to a form I think might work
    #well not work the result is 
    #mse: 9838.962541993975#mae: 81.32000619038823

    test_model_test()
    #use convlstm2d to swap cnn+lstm 
    #mse: 1104.8861090855592#mae: 22.737310008575665
    #convlstm2d to swap cnn+lstm and add dropout=0.1 recurrent_dropout=0.1
    #mse: 1099.6255062036741#mae: 22.729734814561304