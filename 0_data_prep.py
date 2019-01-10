import pandas as pd
import numpy as np

wk_dir     = "D:\\System\\Documents\\ACADEMIC\\HF\\Data\\2018_12_21\\"
file_1     = 'atlantic.csv'
file_1_17  = 'atlantic_2017.csv'
file_2     = 'eastern.csv'
file_2_17  = 'eastern_2017.csv'

col_names  = {'Unnamed: 0':'orig_id'
             ,'V61':'vmax_ivcn'
             ,'V63':'V61_n' 
             ,'V64':'vmax_nhc' 
             ,'V65':'vmax_hwrf'
             ,'V66':'date'    
             ,'V67':'lead_time'
             ,'V68':'orig_storm_id'
             ,'V69':'vmax_op'
             ,'V70':'vmax_navy'
             ,'V71':'vmax_hcca'
             ,'V72':'vmax_hmon'
             ,'V73':'vmax_hwrf_interpolated'
             ,'V74':'vmax'
             ,'V75':'vmax_old'} # need to finish

col_order  = ['storm_id','lead_time','date','vmax','vmax_op',
              'vmax_nhc','vmax_ivcn','vmax_hwrf','orig_storm_id','dataset'
              ,'dataset_ind']


def cube_scaler(x):
    if x == -9999:
        return x
    else:
        # np.log(hf.V62_n+1+np.abs(hf.V62_n.min()))
        return np.sign(x)*np.power(np.abs(x),1/3)   
    

def single_import(filepath,dataset_label,id_num):        
    tmp = pd.read_csv(filepath,dtype={'Unnamed: 0':'int','V68':'int'})
    del tmp['V54']
    tmp.V66 = pd.to_datetime(tmp.V66.astype(str),format='%Y%m%d%H')    
    tmp['storm_id'] = tmp.V66.apply(lambda x: x.year) + (
                        tmp.V68 * 10 + id_num)*10000 # unique storm id 
    
    tmp['dataset'] = dataset_label
    tmp['dataset_ind'] = id_num
    tmp['land_t'] = tmp.V62.apply(cube_scaler)
    for var in ['V6','V8']: # angle variables
        miss = (tmp[var] == -9999).astype(int)
        tmp[var+'_x'] = ((1-miss)*tmp[var].apply(lambda x: np.cos(x*np.pi/180))
                         +miss*(-9999))
        tmp[var+'_y'] = ((1-miss)*tmp[var].apply(lambda x: np.sin(x*np.pi/180))
                         +miss*(-9999))
        del tmp[var]
    
    cols = list(tmp)
    
    # switch IVCN and V61
    a,b=cols.index('V61'),cols.index('V63')
    cols[b],cols[a] = cols[a],cols[b]
    tmp=tmp[cols]
    tmp.rename(columns=col_names, inplace=True)
        
    # t_0 predictions
    predictors = list(tmp)[1:60] + ['land_t','vmax_op','vmax_hwrf'
                     ,'V6_x','V6_y','V8_x','V8_y']
    current= tmp.loc[tmp.lead_time == 0,['storm_id','date']+predictors]
    tmp=tmp.merge(current,on=['storm_id','date']
                ,how='left',suffixes=('','_t0'))

    # missing indicators
    predictors.remove('V6_x')
    predictors.remove('V8_x')
    tmp['n_miss'] = 0
    for ft in predictors + [a+'_t0' for a in predictors]:
        if len(tmp[ft].unique()) == 1:
            del tmp[ft]
        else:
            tmp[ft+'_miss'] = (tmp[ft] == -9999).astype(int)
            tmp['n_miss'] = tmp['n_miss'] + tmp[ft+'_miss']
    # lead time indicators
    for lt in tmp.lead_time.unique():
        tmp['lead_time_'+str(lt)] = (tmp['lead_time'] == lt).astype(int)
    
    tmp = tmp [col_order+[var for var in list(tmp) if var not in col_order]]
    return tmp


atl    = single_import(wk_dir+file_1,'atlantic',0)
atl_17 = single_import(wk_dir+file_1_17,'atlantic',0)
pac    = single_import(wk_dir+file_2,'east_pacific',1)
pac_17 = single_import(wk_dir+file_2_17,'east_pacific',1)
df = atl.append(pac,ignore_index = True)
df = df.append(atl_17,ignore_index = True)
df = df.append(pac_17,ignore_index = True)

#%% RE-INDEX TO MATCH OPERATIONAL PERFORMANCE
response = df.loc[:,['storm_id','date','lead_time',
               'vmax', 'vmax_nhc','vmax_ivcn']]
response['lead_time'] = response['lead_time']+6
response['date'] = response['date']+pd.Timedelta(-6,unit='h')

df = df.merge(response,how='inner',on=['lead_time','date','storm_id'],
              suffixes=('_old',''))

df.to_csv(path_or_buf=wk_dir+'0_clean_data.csv',index=True)
