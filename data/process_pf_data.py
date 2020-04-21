import pandas as pd
import os
terminal_size=os.get_terminal_size()
def extract(file_name:str,route_name:str,direction:str)->pd.DataFrame:
    xls_file=pd.ExcelFile(file_name)
    df=pd.DataFrame()
    csv_file=pd.DataFrame()
    for i in xls_file.sheet_names:
        df=pd.read_excel(xls_file,sheet_name=i,skiprows=2)
        csv_file=csv_file.append(df,ignore_index=True)
    
    file_name=file_name.replace('.xls','.csv')
    file_name=file_name.replace('./data02/data','./pf_data/pf')
    csv_file.columns=['route_name','bus_id','bus_code','card_code','sum_time','longitude','latitude','route_id','sta_id','trip_id','sta_name','direction','sta_order']
    csv_file=csv_file.query('route_name==@route_name and direction==@direction')
    csv_file.to_csv(file_name,index=False)

def extract_2_downflow():
    #extract('./data02/data_20181201_2.xls','2','下行')
    fileprefix='./data02/'
    file_list=os.listdir(fileprefix)
    done_list=os.listdir('./pf_data/')
    counter=len(done_list)
    for i in file_list:
        if i not in done_list:
            extract(fileprefix+i,'2','下行')
            counter+=1
            print(i,' done ',int(counter/len(file_list)*100),'%\r',sep='',end='')
    print('\n extract_2_downflow done')

def merge_pf():
    fileprefix='./pf_data/'
    file_list=os.listdir(fileprefix)
    
    mergefile=pd.DataFrame()
    counter=0
    for i in file_list:
        filepath=fileprefix+i
        part=pd.read_csv(filepath)
        mergefile=mergefile.append(part,ignore_index=True)
        counter+=1
        print('\r',i,' done ',int(counter/len(file_list)*100),'%',sep='',end='')
    #print(' '*terminal_size.columns)
    print('\nmerge done')
    mergefile.to_csv('pf_merge.csv',index=False)

def calculate_pf():
    merged=pd.read_csv('pf_merge.csv')
    merged.sort_values(['sum_time'],inplace=True,ignore_index=True)
    pf_dict=dict()
    counter=0
    for row in merged.iterrows():
        key=str(row[1]['trip_id'])
        key+='%2d' %(row[1]['sta_order'])
        if key in pf_dict.keys():
            pf_dict[key]+=1
        else:
            pf_dict[key]=1
        counter+=1
        print('\r','calculate pf: ',int(counter/len(merged)),'%',sep='',end='')
    print('')
    merged.drop_duplicates(['trip_id','sta_order'],keep='first',inplace=True,ignore_index=True)
    counter=0
    for row in merged.iterrows():
        key=str(row[1]['trip_id'])
        key+='%2d' %(row[1]['sta_order'])
        merged.at[row[0],'pf']=pf_dict[key]
        counter+=1
        print('\r','assign pf: ',int(counter/len(merged)),'%',sep='',end='')
    print('')
    merged.to_csv('pf_calculated.csv',index=False)
    return merged

def make_timeslot_oneday(data:pd.DataFrame,sta_order_start=1,sta_order_end=23,slot_length=15)->pd.DataFrame:
    start=pd.Timedelta(minutes=0)
    maxtime=pd.Timedelta(days=1)
    date_day=data.iloc[0]['sum_time']
    date_day=str(date_day)[:11]
    oneday=pd.DataFrame()
    while start<maxtime:
        start_str=str(start)[-8:]
        start=start+pd.Timedelta(minutes=slot_length)
        end_str=str(start)[-8:]
        oneslot=data.between_time(start_str,end_str,include_end=False)
        sta_sum=dict()
        sta_count=dict()
        for row in oneslot.iterrows():
            so='%02d' %(int(row[1]['sta_order']))
            if so in sta_sum.keys():
                sta_sum[so]+=row[1]['pf']
                sta_count[so]+=1
            else:
                sta_sum[so]=row[1]['pf']
                sta_count[so]=1
        #for i in sta_sum.keys():
        #    sta_sum[i]=round(sta_sum[i]*1.0/sta_count[i],2)
        if len(sta_sum.keys())>0:
            sta_sum['start_time']=date_day+start_str
            oneday=oneday.append(sta_sum,ignore_index=True)
        print('\r\twork done:',date_day+start_str,end='')
    start_time=oneday.pop('start_time')
    oneday.insert(0,'start_time',start_time)
    return oneday

def make_timeslot(data:pd.DataFrame,slot_length=15)->pd.DataFrame:
    data.drop(columns=['bus_id','bus_code','card_code','longitude','latitude','route_id','sta_id'],inplace=True)
    data.set_index(['sum_time'],inplace=True,drop=False)
    last_index=pd.to_datetime('2019-03-21')
    timeslot=pd.DataFrame()
    day_start=pd.to_datetime('2018-12-01')
    day_end=day_start+pd.Timedelta(days=1)
    while day_start<last_index:
        day_end=day_start+pd.Timedelta(days=1)
        oneday=data.query('sum_time>=@day_start and sum_time <@day_end')
        day_start=day_end
        if (len(oneday)!=0):
            oneday=make_timeslot_oneday(oneday)
            timeslot=timeslot.append(oneday,ignore_index=True)
    print('')
    timeslot.sort_index(axis=1,inplace=True)
    timeslot.fillna(value=0,inplace=True)
    start_time=timeslot.pop('start_time')
    timeslot.insert(0,'start_time',start_time)
    timeslot.to_csv('pf_timeslot.csv',index=False)
    return timeslot

if __name__ == "__main__":
    print('process_data')
    #extract_2_downflow()
    #merge_pf()
    #calculate_pf()
    data=pd.read_csv('pf_calculated.csv',parse_dates=['sum_time'])
    data=make_timeslot(data)
    data.to_csv('pf_dataset.csv',index=False)
