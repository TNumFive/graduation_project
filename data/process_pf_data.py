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
if __name__ == "__main__":
    print('process_data')
    #extract_2_downflow()
    #merge_pf()
    calculate_pf()
    
