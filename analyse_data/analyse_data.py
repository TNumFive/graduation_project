import numpy as np
import pandas as pd
import datetime
import os
import sys

from matplotlib import pyplot
from sklearn.metrics import mean_squared_error,mean_absolute_error

def mean_absolute_percentage_error(y_true, y_pred): 
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

np.random.seed(seed=5)

def make_timeslot_oneday(data:pd.DataFrame,slot_length=15)->pd.DataFrame:
    start=pd.Timedelta(minutes=0)
    maxtime=pd.Timedelta(days=1)
    date_day=data.iloc[0]['arrival']
    date_day=str(date_day)[:11]#stringize and slice 0:11#获得日期信息
    #start=str(start).replace('0 days ','')
    oneday=pd.DataFrame()
    while start<maxtime:
        start_str=str(start)[-8:]#从00：00：00开始 00：00：00结束
        start=start+pd.Timedelta(minutes=slot_length)#一个时隙时间长度
        end_str=str(start)[-8:]
        #obtain one slot of data
        oneslot=data.between_time(start_str,end_str,include_end=False)
        sta_sum=dict()
        sta_count=dict()
        for row in oneslot.iterrows():
            so='%02d' %(int(row[1]['sta_order']))
            if so in sta_sum.keys():
                sta_sum[so]+=row[1]['sta_time']
                sta_count[so]+=1
            else:
                sta_sum[so]=row[1]['sta_time']
                sta_count[so]=1
        for i in sta_sum.keys():#用平均法计算时隙内站点时间,round 控制小数位数
            sta_sum[i]=round(sta_sum[i]*1.0/sta_count[i],2)
        if len(sta_sum.keys())>0:#该时隙没有key代表该时隙为空
            sta_sum['start_time']=date_day+start_str#添加时隙标签
            oneday=oneday.append(sta_sum,ignore_index=True)
        print('\twork done:',date_day+start_str,'\r',end='')
    #print(' ')
    return oneday

def make_timeslot(data:pd.DataFrame,slot_length=15,savepath='tt_timeslot.csv')->pd.DataFrame:
    print('make time slot')
    #data.columns='route_id,route_name,sta_id,sta_name,arrival,trip_id,inout,direction,sta_order'
    data.drop(columns=['route_name','sta_id','direction'],inplace=True)
    data.set_index(['arrival'],inplace=True,drop=False)
    #last_index=data.iloc[-1]['arrival']#获取最后一条信息的到达时间
    last_index=pd.to_datetime('2019-03-21')#由于缺少信息，3月份数据只到3-20为止
    timeslot=pd.DataFrame()
    day_start=pd.to_datetime('2018-12-01')#need changes if try to fit other month#遍历每一天
    day_end=day_start+pd.Timedelta(days=1)
    #timeslot=data.query('arrival>= @start and arrival < @end')
    while day_start<last_index:#分批处理每一天的数据
        day_end=day_start+pd.Timedelta(days=1)
        oneday=data.query('arrival>=@day_start and arrival <@day_end')
        day_start=day_end
        if(len(oneday)!=0):
            oneday=make_timeslot_oneday(oneday,slot_length=slot_length)#获得当天的timeslot信息
            timeslot=timeslot.append(oneday,ignore_index=True)
    print('')
    timeslot.sort_index(axis=1,inplace=True)
    start_time=timeslot.pop('start_time')#调整时间位置
    timeslot.insert(0,'start_time',start_time)
    timeslot.to_csv(savepath,index=False)
    return timeslot

if False:#the original tt_add_travel_time.csv contains some data we don't need
    d181201=pd.to_datetime('2018-12-01 00:00:00')
    d181231=pd.to_datetime('2018-12-31 00:00:00')
    d190111=pd.to_datetime('2019-01-11 00:00:00')
    d190321=pd.to_datetime('2019-03-21 00:00:00')
    data=pd.read_csv('tt_add_travel_time.csv',parse_dates=['arrival'])
    data.query('(arrival>=@d181201 and arrival<@d181231) or (arrival>=@d190111 and arrival<@d190321)',inplace=True)
    #by the way just take out the error data like value below or equal to 0
    data.index=range(len(data))
    drop_list=[]
    for i in range(0,len(data)):
        if data.at[i,'sta_time']<=0:
            print(data.at[i,'arrival'])
            drop_list.append(i)
    data.drop(index=drop_list,inplace=True)
    data.to_csv('tt_add_tt_2.csv',index=False)
    if d181201==d181231 and d190111==d190321:
        print('')

def timeslot_analyse(data:pd.DataFrame,sta_order_start=2,sta_order_end=23,slot_length=15):
    #sta_num=sta_order_end-sta_order_start+1
    #print(sta_num)
    index_data=data.loc[:,'start_time']
    data.set_index(['start_time'],inplace=True)
    index_slot:pd.Series=index_data.apply(lambda x: str(x)[-8:])
    index_slot.sort_values(inplace=True,ignore_index=True)
    index_slot=(index_slot[0],index_slot[len(index_slot)-1])
    point=pd.to_datetime(index_slot[0])
    slot_num_max=0
    slot_num_loss_dict=dict()
    slot_sta_loss_dict=dict()
    while point<=pd.to_datetime(index_slot[1]):
        point_str=str(point)[-8:]
        point+=pd.Timedelta(minutes=slot_length)
        slot=data.at_time(point_str)
        len_slot=len(slot)
        slot_num_loss_dict[point_str]=len_slot
        slot_sta_loss_dict[point_str]=round((1-slot.count()/len_slot)*100,2)#for every slot how many sta's data is loss showed in percentage
        if len_slot>slot_num_max:
            slot_num_max=len_slot
    for i in slot_num_loss_dict.keys():#for every specific time ,how many slot was lost 
        slot_num_loss_dict[i]=round(
            (1-slot_num_loss_dict[i]/slot_num_max)*100,2
        )
    sta_loss:dict=data.count().to_dict()
    average_sta_loss=0
    sta_max=0
    for i in sta_loss.keys():
        if sta_loss[i]>sta_max:
            sta_max=sta_loss[i]
    for i in sta_loss.keys():
        sta_loss[i]=round(
            (1-sta_loss[i]/sta_max)*100,2
        )
        average_sta_loss+=sta_loss[i]
    average_sta_loss=average_sta_loss/len(sta_loss)
    savepath='tt_pred_by_timeslot_%02d.csv' % slot_length
    filelist=os.listdir('./')
    if savepath not in filelist:
        raw=pd.read_csv('tt_add_tt_2.csv',parse_dates=['arrival'])
        raw=pd.DataFrame(raw,columns=['arrival','sta_order','sta_time'])
        ts=0
        for i in range(0,len(raw)):
            arrival=raw.at[i,'arrival']
            sta_ord='%02d' % (raw.at[i,'sta_order'])
            print('\r\tmathcing data for sta order:',sta_ord,'arrival:',arrival,end='')
            arrival=arrival-pd.Timedelta(minutes=slot_length)
            while arrival>=index_data[ts]:
                ts=ts+1
            raw.at[i,'pred']=data.at[index_data[ts],sta_ord]
        print('')
        raw.to_csv(savepath,index=False)
    else:
        raw=pd.read_csv(savepath,parse_dates=['arrival'])
    y_true=raw.loc[:,'sta_time'].to_numpy()
    y_pred=raw.loc[:,'pred'].to_numpy()
    mse='mse:%.4f'% mean_squared_error(y_true,y_pred)
    mae='mae:%.4f'% mean_absolute_error(y_true,y_pred)
    mape='mape:%.4f'% mean_absolute_percentage_error(y_true,y_pred) +'%'
    average_sta_loss='average sta loss rate:%.4f'% average_sta_loss +'%'
    print(mse,mae,mape,average_sta_loss)
    
if False:
    for slot_length in possible_slot_length:
        print('make time slot with slot length:',slot_length)
        data=pd.read_csv('tt_add_tt_2.csv',parse_dates=['arrival'])
        savepath='tt_timeslot_%02d.csv' % slot_length
        make_timeslot(data,slot_length=slot_length,savepath=savepath)

if True:
    possible_slot_length=[6,10,12,15,20,30,60]
    filelist=os.listdir('./')
    for slot_length in possible_slot_length:
        print('when slot length is :',slot_length,'mins')
        savepath='tt_timeslot_%02d.csv' % slot_length
        if savepath not in filelist:
            print('make time slot with slot length:',slot_length)
            data=pd.read_csv('tt_add_tt_2.csv',parse_dates=['arrival'])
            make_timeslot(data,slot_length=slot_length,savepath=savepath)
        data=pd.read_csv(savepath,parse_dates=['start_time'])
        timeslot_analyse(data,slot_length=slot_length)


def plot_line(sta_order=(4,23)):
    data=pd.read_csv('tt_dataset.csv',parse_dates=['start_time'])
    raw=pd.read_csv('tt_dataset.csv',parse_dates=['start_time'])
    #make a clean to delete abnormal data
    len_raw=len(raw)
    multiplier=2
    updated=1
    counter=0
    while updated!=0:
        updated=0
        for i in range(4,24):
            i_str='%02d' %i
            for j in range(0,len_raw,72):
                for k in range(1,71):
                    average=(raw.at[j+k-1,i_str]+raw.at[j+k+1,i_str])/2
                    if raw.at[j+k,i_str]>=average*multiplier or raw.at[j+k,i_str]<=average/multiplier:
                        raw.at[j+k,i_str]=average
                        updated=1
        counter+=1
        print('\r\t round :',counter,end='')
    print('')

    for i in range(sta_order[0],sta_order[1]+1):
        i_str='%02d' %i
        average=data.loc[:,i_str].mean()
        pyplot.figure(figsize=(84,18))
        pyplot.plot(data.loc[:,i_str],'-')
        pyplot.plot(raw.loc[:,i_str],':')
        pyplot.plot([0,len(data)],[average,average])
        pyplot.show()
        pyplot.savefig('plot_tt/sta_order_'+i_str+'.png')
