import pandas as pd
import numpy as np
import os
from matplotlib import pyplot

#pay attention to the encoding of inout station data
def data_process(filename:str,zhandian:pd.DataFrame,index_zhandian:np.ndarray)->dict:
    
    inoutstation=pd.read_csv(filename,encoding='gbk',parse_dates=['到站时间'],dtype={'站点ID':np.int32})
    inoutstation.columns=['index','route_id','route_name','car_id','sta_id','sta_name','longitute','latitude','arrival','trip_id','inout']
    inoutstation=pd.DataFrame(inoutstation)
    inoutstation.drop(columns=['index','car_id','longitute','latitude'],inplace=True)
    inoutstation.sort_values(['arrival'],inplace=True,ignore_index=True)
    inoutstation.drop_duplicates(subset=['trip_id','sta_id'],keep='first',inplace=True,ignore_index=True)

    inoutstation.loc[:,'direction']='##'
    inoutstation.loc[:,'sta_order']='##'
    unknown_staid=dict()
    len_inoutstation=len(inoutstation)
    for i in range(0,len_inoutstation):
        sta_id=inoutstation.at[i,'sta_id']
        if sta_id in index_zhandian:
            inoutstation.at[i,'direction']=zhandian.at[sta_id,'direction']
            inoutstation.at[i,'sta_order']=zhandian.at[sta_id,'sta_order']
        else :
            if sta_id in unknown_staid.keys():
                unknown_staid[sta_id]+=1
            else:
                unknown_staid[sta_id]=1
                if inoutstation.at[i,'route_name']=='2路':
                    unknown_staid[sta_id]=999000000
            inoutstation.drop(index=i,inplace=True)
        print('        ',i+1,'/',len_inoutstation,'\r',end='')
    print('')

    filename=filename.replace('inout-station','add_order_')
    inoutstation.to_csv(filename,index=False)
    return unknown_staid

if __name__ == "__main__":
    zhandian=pd.read_csv('zhandian20190327.csv')
    zhandian.columns=['sta_id','sta_name','direction','route_id','route_name','sta_order','sta_distance','longitude','latitude']
    zhandian=pd.DataFrame(zhandian,
                        columns=['sta_id','direction','sta_order'])
    zhandian=zhandian.append({'sta_id':11914,'direction':'下行','sta_order':21},ignore_index=True)
    zhandian=zhandian.append({'sta_id':21728,'direction':'上行','sta_order':1},ignore_index=True)

    index_zhandian=zhandian.loc[:,'sta_id'].to_numpy()
    zhandian.set_index(keys=['sta_id'],inplace=True)

    unknown_staid_intotal=dict()
    dl=os.listdir('./')
    fn='inout-station201812_##.csv'
    for i in range(1,31):
        filename=fn.replace('##',str(i))
        if filename in dl:
            print('working on',filename)
            unknown_staid=data_process(filename,zhandian,index_zhandian)
            print('\t',unknown_staid)
            for k in unknown_staid.keys():
                if k in unknown_staid_intotal.keys():
                    unknown_staid_intotal[k]+=unknown_staid[k]
                else:
                    unknown_staid_intotal[k]=unknown_staid[k]
    print('in total')
    for key in unknown_staid_intotal.keys():
        print(unknown_staid_intotal,':',unknown_staid_intotal[key])  