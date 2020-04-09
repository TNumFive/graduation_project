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
    weather,windd=onehot_external_features()
    print(weather)
    print(windd)