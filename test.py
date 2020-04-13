import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

def onehot_external_features()->(dict,dict):
    ef=pd.read_csv('./data/weather.csv',parse_dates=['date'])
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
    weather_array=np.array(weather_list).reshape(-1,1)
    wd_array=np.array(wd_list).reshape(-1,1)
    weather_array=enc.fit_transform(weather_array).toarray()
    wd_array=enc.fit_transform(wd_array).toarray()
    weather_dict=dict()
    for i in range(0,len(weather_array)):
        weather_dict[weather_list[i]]=weather_array[i]
    wd_dict=dict()
    for i in range(0,len(wd_array)):
        wd_dict[wd_list[i]]=wd_array[i]
    return weather_dict,wd_dict
if __name__ == "__main__":
    from keras.models import Sequential,load_model
    from keras.layers import BatchNormalization,LSTM,Dropout,Dense,RepeatVector,Flatten,TimeDistributed
    from keras.utils import print_summary

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

    