import numpy as np
import pandas as pd
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
    reduce_lr=ReduceLROnPlateau()
    callback_list=[earlystop,reduce_lr]
    model.fit(x_train,y_train,batch_size=16,epochs=2000,validation_data=[x_test,y_test],callbacks=callback_list)
    model.save_weights('stdn_weight.h5')
    mms:MinMaxScaler=data['mms']
    y_pred=model.predict(x_test)
    y_true=mms.inverse_transform(y_test[0].flatten().reshape(-1,1)).flatten()
    y_pred=mms.inverse_transform(y_pred.flatten().reshape(-1,1)).flatten()
    print('mse:',mean_squared_error(y_true,y_pred))
    print('mae:',mean_absolute_error(y_true,y_pred))


if __name__ == "__main__":
    print('comparison.py')
    stdn_like_model_test()
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
