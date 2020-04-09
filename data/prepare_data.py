import numpy as np
import pandas as pd 
import os
import datetime
from matplotlib import pyplot

#汇总目标路线指定运营方向的数据
def merge()->pd.DataFrame:
    data=pd.DataFrame()#merge data from 201812 201901 201903
    data=data.append(pd.read_csv('temp_merge1812.csv',parse_dates=['arrival']),ignore_index=False)
    data=data.append(pd.read_csv('temp_merge1901.csv',parse_dates=['arrival']),ignore_index=False)
    data=data.append(pd.read_csv('temp_merge1903.csv',parse_dates=['arrival']),ignore_index=False)
    return data

def add_triptime(data:pd.DataFrame)->pd.DataFrame:
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
        print('\tadd trip time : %d/%d' %(i,len(data)),'\r',end='')
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
        print('\tadd sta time : %d/%d' %(i,len(data)),'\r',end='')
    print(' ')
    data.sort_values(['arrival'],inplace=True,ignore_index=True)
    data.query('sta_time != -1',inplace=True)
    return data

def within_oneday(data:pd.DataFrame,sta_order_start=1,sta_order_end=23,slot_length=15)->pd.DataFrame:
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
        for i in sta_sum.keys():#用平均法计算时隙内站点时间
            sta_sum[i]=round(sta_sum[i]*1.0/sta_count[i],2)
        if len(sta_sum.keys())>0:#该时隙没有key代表该时隙为空
            sta_sum['start_time']=date_day+start_str#添加时隙标签
            oneday=oneday.append(sta_sum,ignore_index=True)
        print('\tworking on:',date_day+start_str,'\r',end='')
    #print(' ')
    start_time=oneday.pop('start_time')#调整时间位置
    oneday.insert(0,'start_time',start_time)
    return oneday

def make_timeslot(data:pd.DataFrame,slot_length=15)->pd.DataFrame:
    #data.columns='route_id,route_name,sta_id,sta_name,arrival,trip_id,inout,direction,sta_order'
    data.drop(columns=['route_id','route_name','sta_id','inout','direction'],inplace=True)
    data.set_index(['arrival'],inplace=True,drop=False)
    #last_index=data.iloc[-1]['arrival']#获取最后一条信息的到达时间
    last_index=pd.to_datetime('2019-03-21')#由于缺少信息，3月份数据只到3-20为止
    timeslot=pd.DataFrame()
    day_start=pd.to_datetime('2018-12-01')#need changes if try to fit other month#遍历3月的每一天
    day_end=day_start+pd.Timedelta(days=1)
    #timeslot=data.query('arrival>= @start and arrival < @end')
    while day_start<last_index:#分批处理每一天的数据
        day_end=day_start+pd.Timedelta(days=1)
        oneday=data.query('arrival>=@day_start and arrival <@day_end')
        day_start=day_end
        if(len(oneday)!=0):
            oneday=within_oneday(oneday)#获得当天的timeslot信息
            timeslot=timeslot.append(oneday,ignore_index=True)
    print(' ')
    return timeslot

def prepare_dataset(data:pd.DataFrame,input_steps=6,output_steps=3,slot_length=15)->pd.DataFrame:
    #we need to find a way to know a new day has come
    #first, let's test the index use of time
    dataset=pd.DataFrame()
    data.set_index('start_time',inplace=True,drop=False)
    for i in range(input_steps-1,len(data)):
        nowtime=data.iloc[i].name
        #print(nowtime)
        #新建dataframe指定列顺序，注意由于包含‘start time’，列名被识别为str类型，由于采用cnn，列顺序需要固定
        tempslot=pd.DataFrame()
        shortterm=tempslot
        inputslot=tempslot
        outputslot=tempslot
        for j in range(input_steps*-1,0):#短期依赖
            temptime=nowtime+pd.Timedelta(minutes=15*j)
            if temptime in data.index:
                shortterm=shortterm.append(data.loc[temptime,:],ignore_index=True)
            else:
                shortterm=shortterm.append({'start_time':temptime},ignore_index=True)
        for j in range(0,output_steps):#输出步数
            temptime=nowtime+pd.Timedelta(minutes=15*j)
            if temptime in data.index:
                outputslot=outputslot.append(data.loc[temptime,:],ignore_index=True)
            else:
                outputslot=outputslot.append({'start_time':temptime},ignore_index=True)
        
        assert input_steps <=4#fillna is set according to the input_steps
        #here shall be updated to a more suitable way
        if nowtime.hour==23:
            outputslot.fillna(method='ffill',inplace=True)
        if nowtime.hour==5:
            shortterm.fillna(method='bfill',inplace=True)
    
        inputslot=inputslot.append(shortterm,ignore_index=True)

        tempslot=tempslot.append(inputslot,ignore_index=True)
        tempslot=tempslot.append(outputslot,ignore_index=True)
        #tempslot.fillna(value=0,inplace=True)#此处也用0补充
        #tempslot.fillna(method='bfill',inplace=True)
        #计算tempslot中0的个数，特殊情况已经进行了补0，正常情况下站点时间不可能为0，去除含0过多的tempslot
        #isna_sumup=tempslot.isna().sum()
        #loss_total=0
        #for key in isna_sumup:
        #    loss_total+=isna_sumup[key]
        ratio=0.0001#basiclly means "no loss please!"
        if shortterm.isna().sum().sum()<23*(input_steps*1)*ratio:
            if outputslot.isna().sum().sum()<23*(output_steps*1)*ratio:
                if tempslot.isna().sum().sum() < 23*(input_steps+output_steps)*ratio:
                    dataset=dataset.append(tempslot,ignore_index=True)
        print('\rfinish time:',nowtime,end='')
    print('')
    #最终仍有少量nan，则进行补0
    dataset.fillna(value=0,inplace=True)    
    return dataset

def prepare_dataset2(data:pd.DataFrame,fore_step=4,sta_num=22,output_step=1,slot_length=15,ratio=0.05)->(pd.DataFrame,pd.DataFrame,pd.DataFrame,pd.DataFrame):
    st=pd.DataFrame()#short term dependency
    lt=pd.DataFrame()#long term dependency
    se=pd.DataFrame()#short term external features
    le=pd.DataFrame()#long term external
    op=pd.DataFrame()#output data
    data.set_index('start_time',inplace=True,drop=False)
    weather=pd.read_csv('weather.csv',parse_dates=['date'])
    #date,AQI,qg,PM2.5,PM10,SO2,CO,NO2,O3_8h,dw,nw,ht,lt,wd,wf
    weather=pd.DataFrame(weather,columns=['date','AQI','ht','lt','wf'])
    weather.set_index(['date'],inplace=True,drop=False)
    #add weekdayt te weather
    #todo:
    for i,c in weather.iterrows():
        temptime=pd.to_datetime(c['date'])
        wd=datetime.datetime.fromtimestamp(i.timestamp()).weekday()
        weather.at[i,'wday']=wd

    columns=['start_time','02','03','04','05','06','07','08','09','10','11','12','13','14','15','16','17','18','19','20','21','22','23']
    col_ef=['date','AQI','ht','lt','wf','wday']
    for i in range(fore_step-1,len(data)):
        rowtime:pd.Timedelta = data.iloc[i].name
        s=pd.DataFrame(columns=columns)#short term dependency
        l=pd.DataFrame(columns=columns)#long term dependency
        es=pd.DataFrame(columns=col_ef)#external feature
        el=pd.DataFrame(columns=col_ef)
        o=pd.DataFrame(columns=columns)#output
        for j in range(fore_step*-1,0):
            temptime=rowtime+pd.Timedelta(minutes=slot_length*j)
            if temptime in data.index:
                s=s.append(data.loc[temptime,:],ignore_index=True)
            else:
                s=s.append({'start_time':temptime},ignore_index=True)
            temptime=rowtime+pd.Timedelta(days=j)
            if temptime in data.index:
                l=l.append(data.loc[temptime,:],ignore_index=True)
            else:
                l=l.append({'start_time':temptime},ignore_index=True)
            temptime=str(temptime)[:-9]
            temptime=pd.to_datetime(temptime)
            if temptime in weather.index:
                el=el.append(weather.loc[temptime,:],ignore_index=True)
            else:
                el=el.append({'date':temptime},ignore_index=True)
        for j in range(0,output_step):
            temptime=rowtime+pd.Timedelta(minutes=slot_length*j)
            if temptime in data.index:
                o=o.append(data.loc[temptime,:],ignore_index=True)
            else:
                o=o.append({'start_time':temptime},ignore_index=True)
            temptime=str(temptime)[:-9]
            temptime=pd.to_datetime(temptime)
            if temptime in weather.index:
                es=es.append(weather.loc[temptime,:],ignore_index=True)
            else:
                es=es.append({'date':temptime},ignore_index=True)
        #long short term ,one of them fit the loss rate is fine
        threshold=sta_num*fore_step*ratio
        temptime=str(rowtime)[-8:]
        temptime=pd.to_datetime(temptime)
        
        c1=(temptime<=(pd.to_datetime('5:00:00')+pd.Timedelta(minutes=slot_length*fore_step)))
        if c1:
            s.fillna(method='bfill',inplace=True)
            s.fillna(method='ffill',inplace=True)
        if temptime>=(pd.to_datetime('23:30:00')-pd.Timedelta(minutes=slot_length*fore_step)):
            o.fillna(method='ffill',inplace=True)
            o.fillna(method='bfill',inplace=True)
        
        s_isna=s.isna().sum().sum()
        l_isna=l.isna().sum().sum()
        o_isna=o.isna().sum().sum()

        c1=c1 and (l_isna<=threshold)#speical timeslot that long term dependency is adequate
        c2=(l_isna<=threshold) and (s_isna<=threshold)# no loss in long short term dependency
        c3=(o_isna<=0)#no loss is allowed
        
        if c3:
            if c1 or c2:
            #if s_isna<=threshold:#output has no loss and short term has no loss
                st=st.append(s,ignore_index=True)
                lt=lt.append(l,ignore_index=True)
                se=se.append(es,ignore_index=True)
                le=le.append(el,ignore_index=True)
                op=op.append(o,ignore_index=True)
        print('\tfinished ts:',rowtime,c1,c2,c3,threshold,s_isna,l_isna,o_isna,'                     ','\r',end='')
    
    print('\nlength of data:',len(op))
    va=op.mean()#there might be value ==0 /xk
    for index,content in op.iterrows():
        for i in content.keys():
            if content[i]==0.0:
                op.at[index,i]=va[i]
    
    st.fillna(value=va,inplace=True)
    lt.fillna(value=va,inplace=True)
    se.fillna(value=va,inplace=True)
    op.fillna(value=va,inplace=True)
    return st,lt,se,le,op

if __name__ == "__main__":
    #os.chdir('./data')
    '''    
    print('utility , prepare data !!!')
    print('merge()')
    raw=merge()#return data that route=2 and direction =xiaxing
    raw.to_csv('temp_merge.csv',index=False)

    print('add trip time and sta_time')
    raw=pd.read_csv('temp_merge.csv',parse_dates=['arrival'])
    raw=add_triptime(raw)
    raw.to_csv('temp_add_triptime.csv',index=False)

    print('make time slots')
    raw=pd.read_csv('temp_add_triptime.csv',parse_dates=['arrival'])
    raw=make_timeslot(raw)
    raw.to_csv('temp_timeslot.csv',index=False)
    '''
    print('prepare dataset for nn')
    #raw=pd.read_csv('temp_timeslot_fixup.csv',parse_dates=['start_time'])
    raw=pd.read_csv('temp_timeslot.csv',parse_dates=['start_time'])
    raw=prepare_dataset2(raw)
    ln=['st','lt','se','le','op']
    for i in range(0,len(ln)):
        raw[i].to_csv('temp_dataset_'+ln[i]+'.csv',index=False)
        print(ln[i],':',len(raw[i]))

