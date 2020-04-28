import pandas as pd
import numpy as np
import datetime
import os

def extract(file_name:str,route_name:str)->pd.DataFrame:
    inout=pd.read_csv(file_name,encoding='gbk',parse_dates=['到站时间'])
    inout.columns=['index','route_id','route_name','bus_id','sta_id','sta_name','longitude','latitude','arrival','trip_id','inout']
    inout.drop(columns=['index','route_id','bus_id','longitude','latitude','inout'],inplace=True)
    inout.query('route_name==@route_name',inplace=True)
    inout.sort_values(['arrival'],ignore_index=True,inplace=True)
    inout.drop_duplicates(['trip_id','sta_id'],keep='first',inplace=True,ignore_index=True)
    return inout

def extract_2_downflow()->pd.DataFrame:
    fileprefix='./tt_data/'
    file_list=os.listdir(fileprefix)
    counter=0
    r2df=pd.DataFrame()
    for i in file_list:
        part=extract(fileprefix+i,'2路')
        r2df=r2df.append(part,ignore_index=True)
        counter+=1
        print('\r\t',i,' done ',counter,'/',len(file_list),sep='',end='')
    print('\nmerge done')
    r2df.sort_values(['arrival'],ignore_index=True,inplace=True)
    r2df.to_csv('tt_merge.csv',index=False)
    return r2df

def add_order(r2df:pd.DataFrame,direction='下行')->pd.DataFrame:
    zhandian=pd.read_csv('zhandian20190327.csv')
    zhandian.columns=['sta_id','sta_name','direction','route_id','route_name','sta_order','sta_distance','longitude','latitude']
    zhandian=pd.DataFrame(zhandian,columns=['sta_id','direction','sta_order'])
    zhandian=zhandian.append({'sta_id':11914,'direction':'下行','sta_order':21},ignore_index=True)
    zhandian=zhandian.append({'sta_id':21728,'direction':'上行','sta_order':1},ignore_index=True)
    index_zhandian=zhandian.loc[:,'sta_id'].to_numpy()
    zhandian.set_index(keys=['sta_id'],inplace=True)
    
    #r2df=pd.read_csv('tt_merge.csv',parse_dates=['arrival'])
    r2df.loc[:,'direction']='##'
    r2df.loc[:,'sta_order']='##'
    unknown_staid=dict()
    len_r2df=len(r2df)
    for i in range(0,len_r2df):
        sta_id=r2df.at[i,'sta_id']
        if sta_id in index_zhandian:
            r2df.at[i,'direction']=zhandian.at[sta_id,'direction']
            r2df.at[i,'sta_order']=zhandian.at[sta_id,'sta_order']
        else:
            if sta_id in unknown_staid.keys():
                unknown_staid[sta_id]+=1
            else:
                unknown_staid[sta_id]=1
        print('\r\t','add_order:',i+1,'/',len_r2df,sep='',end='')
    print('')
    r2df.query('direction==@direction',inplace=True)
    print('add_order done!')
    r2df.to_csv('tt_add_order.csv',index=False)
    return r2df

def add_travel_time(data:pd.DataFrame)->pd.DataFrame:
    data.sort_values(['sta_order','arrival'],inplace=True,ignore_index=True)
    data.at[0,'trip_time']=0# time between last trip and this trip
    for i in range(1,len(data)):
        if data.at[i-1,'sta_order']==data.at[i,'sta_order']:
            temp=data.at[i,'arrival']-data.at[i-1,'arrival']
            temp=temp.total_seconds()
            if temp>7200:#上下趟时间超过2小时肯定是因为两趟不是同一天
                temp=0
            data.at[i,'trip_time']=temp
        else:#换站点了，本站为新的站点的第一趟，没有上一趟
            data.at[i,'trip_time']=0
        print('\tadd trip time : %d/%d' %(i+1,len(data)),'\r',end='')
    print(' ')
    data.sort_values(['trip_id','sta_order'],inplace=True,ignore_index=True)
    data.at[0,'sta_time']=0# time between last station and this station
    if data.at[0,'sta_order']!=1:#第一个站，站序不为1，没有上一个站，该信息应被舍去
        data.at[0,'sta_time']=-1
    for i in range(1,len(data)):
        if data.at[i-1,'trip_id']==data.at[i,'trip_id']:#same trip
            if data.at[i-1,'sta_order']+1==data.at[i,'sta_order']:#last sta
                temp=temp=data.at[i,'arrival']-data.at[i-1,'arrival']
                temp=temp.total_seconds()
                data.at[i,'sta_time']=temp
            else:
                data.at[i,'sta_time']=-1
        else:#different trip
            '''
            if data.at[i,'sta_order']==1:
                data.at[i,'sta_time']=data.at[i,'trip_time']#使用趟次时间代替
            else:
                data.at[i,'sta_time']=-1#数据缺失，待舍去
            '''
            data.at[i,'sta_time']=-1#不论是站序为1，还是数据缺失，均采取舍去
        print('\tadd sta time : %d/%d' %(i+1,len(data)),'\r',end='')
    print('')
    data.sort_values(['arrival'],inplace=True,ignore_index=True)
    data.query('sta_time != -1',inplace=True)
    data.to_csv('tt_add_travel_time.csv',index=False)
    return data

def make_timeslot_oneday(data:pd.DataFrame,sta_order_start=1,sta_order_end=23,slot_length=15)->pd.DataFrame:
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

def make_timeslot(data:pd.DataFrame)->pd.DataFrame:
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
            oneday=make_timeslot_oneday(oneday)#获得当天的timeslot信息
            timeslot=timeslot.append(oneday,ignore_index=True)
    print('')
    timeslot.sort_index(axis=1,inplace=True)
    start_time=timeslot.pop('start_time')#调整时间位置
    timeslot.insert(0,'start_time',start_time)
    timeslot.to_csv('tt_timeslot.csv',index=False)
    return timeslot

def timeslot_analyse(data:pd.DataFrame,sta_order_start=1,sta_order_end=23,slot_length=15):
    start=pd.Timedelta(minutes=0)
    maxtime=pd.Timedelta(days=1)
    data.set_index(['start_time'],drop=False,inplace=True)
    while start<maxtime:
        start_str=str(start)[-8:]#从00：00：00开始 00：00：00结束
        start=start+pd.Timedelta(minutes=slot_length)#一个时隙时间长度
        slot=data.at_time(start_str)
        if len(slot)!=0:
            loss=slot.isna().sum().sum()
            total=slot.fillna(value=0).to_numpy().flatten()
            total=len(total)
            rate=loss/total*100.0
            if loss!=0:
                print(start_str,' ','loss rate: %4d/%4d %5.2f' %(loss,total,rate),'% total:',len(slot),sep='')
    for i in range(sta_order_start,sta_order_end+1):
        slot=data.loc[:,'%02d' %i]
        loss=slot.isna().sum().sum()
        total=slot.fillna(value=0).to_numpy().flatten()
        total=len(total)
        rate=loss/total*100.0
        if loss!=0:
            print('sta_order:','%02d' %i,' loss rate:%02f'%rate,sep='')
    #去除站序1，2，3，选择时隙05：15：00-23：00：00

def prepare_dataset(data:pd.DataFrame,slot_length=15):
    data.set_index(['start_time'],drop=False,inplace=True)
    #先尝试对数据进行补全#补全的目的是为了让数据可用，减少丢失数据对模型的不利的影响,在不清楚的情况下，我直接前后传递好了
    data.fillna(method='ffill',inplace=True)
    data.fillna(method='bfill',inplace=True)
    #去除站序1，2，3，选择时隙05：15：00-23：00：00
    #选择12月1号到30号为止，1月11号开始3月20号为止的数据,因为其他时段缺少客流数据
    data.drop(columns=['02','03'],inplace=True)
    d181201=pd.to_datetime('2018-12-01 05:15:00')
    d181230=pd.to_datetime('2018-12-30 23:00:00')
    d190111=pd.to_datetime('2019-01-11 05:15:00')
    d190131=pd.to_datetime('2019-01-31 23:00:00')
    d190301=pd.to_datetime('2019-03-01 05:15:00')
    d190320=pd.to_datetime('2019-03-20 23:00:00')
    data.query('(start_time>=@d181201 and start_time<=@d181230) or (start_time>=@d190111 and start_time<=@d190320)',inplace=True)
    data=data.between_time('05:15:00','23:00:00')
    #补全缺失的那两个时隙（可能不在可用的时间范围内/xk），以及其他时隙缺失的少量数据
    index_data=data.loc[:,'start_time']
    daysta=d181201
    while daysta<=d181230:
        time=daysta
        dayend=str(daysta).replace('05:15:00','23:00:00')
        dayend=pd.to_datetime(dayend)
        daysta=daysta+pd.Timedelta(days=1)
        while time<=dayend:
            if time not in index_data:
                data=data.append({'start_time':time},ignore_index=True)
                #print(time)
            time+=pd.Timedelta(minutes=slot_length)
    daysta=d190111
    while daysta<=d190131:
        time=daysta
        dayend=str(daysta).replace('05:15:00','23:00:00')
        dayend=pd.to_datetime(dayend)
        daysta=daysta+pd.Timedelta(days=1)
        while time<=dayend:
            if time not in index_data:
                data=data.append({'start_time':time},ignore_index=True)
                #print(time)
            time+=pd.Timedelta(minutes=slot_length)
    daysta=d190301
    while daysta<=d190320:
        time=daysta
        dayend=str(daysta).replace('05:15:00','23:00:00')
        dayend=pd.to_datetime(dayend)
        daysta=daysta+pd.Timedelta(days=1)
        while time<=dayend:
            if time not in index_data:
                data=data.append({'start_time':time},ignore_index=True)
                #print(time)
            time+=pd.Timedelta(minutes=slot_length)
    data.sort_values(['start_time'],inplace=True,ignore_index=True)
    #补全时隙后，仍然有数据'空洞'出现，说明数据缺失并不平均，
    #但三每个时隙的总缺失率并不高，说明缺失集中在某个时隙附近，而且这些时隙并不固定，不然单个时隙的总缺失率会很高
    #数据补全应该在去除数据之前就做处理,此处仅为补全补进去的时隙
    data.fillna(method='ffill',inplace=True)
    data.fillna(method='bfill',inplace=True)
    #以防数据异常，存在0值或小于0的值
    for r in data.iterrows():
        for i in range(1,len(r[1])):
            if r[1][i]<=0:
                data.at[r[0],r[1].keys()[i]]=data.iloc[r[0]-1][r[1].keys()[i]]
    data.to_csv('tt_dataset.csv',index=False)
    #现在处理成与客流数据相对应，且无缺失，所有时隙都在的数据了/xk
    #处理一下完整版的multi-output和STDN模型
    return data
        
def analyse(data:pd.DataFrame,sta_order_start=4,sta_order_end=23):
    data.sort_values(['arrival'],inplace=True,ignore_index=True)
    data.query('sta_order>=@sta_order_start',inplace=True)
    data.query('sta_order<=@sta_order_end',inplace=True)
    sta=data.drop_duplicates(['trip_id'],keep='first',ignore_index=True)
    end=data.drop_duplicates(['trip_id'],keep='last',ignore_index=True)
    total=pd.Timedelta(minutes=0)
    counter=0
    for i in range(0,len(sta)):
        if sta.at[i,'sta_order']==sta_order_start and end.at[i,'sta_order']==sta_order_end:
            total+=(end.at[i,'arrival']-sta.at[i,'arrival'])
            counter+=1
        print('\r\tworking on:',i,'/',len(sta),end='')
    print('')
    print('average time for one trip is :',total/counter)

if __name__ == "__main__":
    print('process_tt_data')
    '''
    data=extract_2_downflow()
    data=add_order(data)
    data=add_travel_time(data)
    data=pd.read_csv('tt_add_travel_time.csv',parse_dates=['arrival'])
    data=make_timeslot(data)
    
    data=pd.read_csv('tt_timeslot.csv',parse_dates=['start_time'])
    #timeslot_analyse(data,sta_order_start=2)
    data=prepare_dataset(data)
    timeslot_analyse(data,sta_order_start=4)
    '''
    data=pd.read_csv('tt_add_order.csv',parse_dates=['arrival'])
    analyse(data)