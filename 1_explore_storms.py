import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

### PROGRAM PARAMETERS ###
wk_dir     = "D:\\System\\Documents\\ACADEMIC\\HF\\Data\\2018_05_11\\"
filename   = '0_clean_data.csv'
core_vars  = ['storm_id','lead_time','date','vmax','vmax_ivcn','vmax_pred'
              ,'vmax_hwrf']

storms_miss = [100,130,140,571,330,481,90,50,221,431,591]
storms_nonmiss = [51,391,601,621,631,10,40,70]
                 
### FUNCTIONS ###
def storm_curves_fix_date(df,storm_list,competitors):
    for storm_id in storm_list:
        rows = df.storm_id == storm_id
        tmp=df.loc[(rows) & (df.date == df[rows].date.min())
                    ,['lead_time','vmax']+['vmax_'+c for c in competitors]]
        if len(tmp) >1 :
            tmp.plot(x='lead_time',title='Storm '+str(storm_id)
                +'\nVMAX and estimates from first prediction')
        else : print('Too few obs: storm '+str(storm_id))

def dataset_compare(df,competitors):
    model_perf = df.groupby(['lead_time','dataset'])[
            ['abs_err_'+c for c in competitors]].agg('mean').reset_index()
    fig = plt.figure()
    ax1 = fig.add_subplot(221)
    ax2 = fig.add_subplot(222)
    model_perf[model_perf.dataset == 'atlantic'].plot(x='lead_time',ax=ax1
              ,xticks=model_perf.lead_time.unique()
              ,title='Atlantic',sharey=True,figsize=(11,8))
    model_perf[model_perf.dataset == 'east_pacific'].plot(x='lead_time',ax=ax2
              ,xticks=model_perf.lead_time.unique(),sharey=True,title='East Pacific')
    return model_perf.sort_values(['dataset','lead_time'])

df = pd.read_csv(wk_dir+filename,index_col=0,parse_dates=['date']) 
df.replace(-9999,value=np.NaN,inplace=True)
        
storm_curves_fix_date(df,storms_nonmiss,['op_t0'])

#%% Missing data


df = pd.read_csv(wk_dir+filename,index_col=0,parse_dates=['date']) 
df.replace(-9999,value=np.NaN,inplace=True)
df=df.loc[df.date < '2017-01-01','storm_id':'V62']

df=pd.DataFrame(columns=[df.isnull().mean(),df.isnull().count()]) 