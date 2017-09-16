import pandas as pd

wk_dir     = "D:\\System\\Documents\\ACADEMIC\\HF\\Data\\"
file_1     = 'atlantic_dataset.csv'
file_2     = 'eastern_dataset.csv'

col_names  = {'Unnamed: 0':'orig_id'
             ,'V61':'vmax_ivcn'
             ,'V63':'V61_n' 
             ,'V64':'vmax_nhc' 
             ,'V65':'vmax_hwrf'
             ,'V66':'date'    
             ,'V67':'lead_time'
             ,'V68':'orig_storm_id'
             ,'V69':'vmax'}

col_order  = ['storm_id','lead_time','date','vmax_t0','vmax','vmax_nhc',
              'vmax_ivcn','vmax_hwrf','orig_storm_id','dataset'
              ,'dataset_ind']

def single_import(filepath,dataset_label,id_num):        
    tmp = pd.read_csv(filepath,dtype={'Unnamed: 0':'int','V68':'int'})
    del tmp['V54']
    tmp.V66 = pd.to_datetime(tmp.V66.astype(str),format='%Y%m%d%H')    
    tmp['storm_id'] = tmp.V68 * 10 + id_num # unique storm id 
    tmp['dataset'] = dataset_label
    tmp['dataset_ind'] = id_num
    cols = list(tmp)
    
    # switch IVCN and V61
    a,b=cols.index('V61'),cols.index('V63')
    cols[b],cols[a] = cols[a],cols[b]
    tmp=tmp[cols]
    tmp.rename(columns=col_names, inplace=True)
        
    predictors = list(tmp)[1:63] + ['vmax','vmax_hwrf']
    current= tmp.loc[tmp.lead_time == 0,['storm_id','date']+predictors]
    tmp=tmp.merge(current,on=['storm_id','date']
                ,how='left',suffixes=('','_t0'))

    for ft in predictors + [a+'_t0' for a in predictors]:
        if len(tmp[ft].unique()) == 1:
            del tmp[ft]
        else:
            tmp[ft+'_miss'] = (tmp[ft] == -9999).astype(int)
    
    tmp = tmp [col_order+[var for var in list(tmp) if var not in col_order]]
    return tmp

atl = single_import(wk_dir+file_1,'atlantic',0)
pac = single_import(wk_dir+file_2,'east_pacific',1)
df = atl.append(pac,ignore_index = True)

df.to_csv(path_or_buf=wk_dir+'0_clean_data.csv',index=True)
