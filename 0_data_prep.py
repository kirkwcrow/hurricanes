import pandas as pd
import numpy as np

wk_dir     = "D:\\System\\Documents\\ACADEMIC\\HF\\Data\\2017_12_06\\"
file_1     = 'atlantic_dataset_adecks_smooth.csv'
file_2     = 'eastern_dataset_adecks_smooth.csv'

col_names  = {'Unnamed: 0':'orig_id'
             ,'V61':'vmax_ivcn'
             ,'V63':'V61_n' 
             ,'V64':'vmax_nhc' 
             ,'V65':'vmax_hwrf'
             ,'V66':'date'    
             ,'V67':'lead_time'
             ,'V68':'orig_storm_id'
             ,'V69':'vmax_op'
             ,'V70':'vmax'}

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
    tmp['storm_id'] = tmp.V68 * 10 + id_num # unique storm id 
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

atl = single_import(wk_dir+file_1,'atlantic',0)
pac = single_import(wk_dir+file_2,'east_pacific',1)
df = atl.append(pac,ignore_index = True)

df.to_csv(path_or_buf=wk_dir+'0_clean_data.csv',index=True)
