import numpy as np
import pandas as pd 
import os
import datetime
from matplotlib import pyplot

#汇总目标路线指定运营方向的数据
def merge(route_name='2路',direction='下行')->pd.DataFrame:
    filename=str('./add_order_201812_##.csv')
    data=pd.DataFrame()
    for i in range(1,31):
        fn=filename.replace('##',str(i))
        print(fn,'starting\r',sep=' ',end='')
        raw=pd.DataFrame(pd.read_csv(fn,parse_dates=['arrival'],dtype={'sta_order':np.int16}))
        raw.query('route_name == "'+route_name+'"',inplace=True)
        raw.query('direction == "'+direction+'"',inplace=True)
        data=pd.concat([data,raw])
        print('        ',fn,'done!!!')
    data.sort_values(['arrival'],inplace=True)#按时间排序，由于多个文件间有重复数据，排序后保留新数据
    data.drop_duplicates(subset=['trip_id','sta_order'],keep='last',inplace=True,ignore_index=True)
    return data


if __name__ == "__main__":
    #os.chdir('./data')

    print('utility , prepare data !!!')
    print('merge()')
    raw=merge()#return data that route=2 and direction =xiaxing
    raw.to_csv('temp_merge12.csv',index=False)
