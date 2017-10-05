import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

wk_dir     = "D:\\System\\Documents\\ACADEMIC\\HF\\Data\\"
filename   = '1_model_preds.csv' #'0_clean_data.csv'
core_vars  = ['storm_id','lead_time','date','vmax','vmax_ivcn','vmax_pred'
              ,'vmax_hwrf']
vmax_vals  = ['vmax','vmax_ivcn','vmax_pred','vmax_hwrf']

storms_miss = [100,130,140,571,330,481,90,50,221,431,591]
storms_nonmiss = [51,391,601,621,631,10,40,70]
                 
df = pd.read_csv(wk_dir+filename,index_col=0,parse_dates=['date']) 
df.replace(-9999,value=np.NaN,inplace=True)

#for storm_id in storms_nonmiss:
#    rows = df.storm_id == storm_id
#    tmp=df.loc[(rows) & (df.date == df[rows].date.min()),['lead_time']+vmax_vals]
#    tmp.plot(x='lead_time',xlim=[3,24],xticks=[3,6,9,12,15,18,21,24],
#             title='Storm '+str(storm_id)
#            +'\nVMAX and estimates from first prediction')

#storms = df.groupby(['storm_id','lead_time']).agg(
#        {'date':['count','min','max'],'vmax':['min','mean','max','count']
#        ,'vmax_hwrf':['count'],'vmax_pred':['count'],'vmax_ivcn':['count']})
#storms.columns = ['_'.join(col).strip() for col in storms.columns.values]
#storms['duration'] = (storms[('date_max')] - storms[('date_min')]).dt.days
#for v in vmax_vals:
#    storms[v+'_miss'] = storms['date_count']-storms[v+'_count']
#    del storms[v+'_count']
#del storms['date_max']
#storms.columns = ['n','start_dt'] + list(storms)[2:]
#storms.to_clipboard()


lead_time = 3
for storm_id in storms_nonmiss:
    rows = df.storm_id == storm_id
    tmp=df.loc[(rows) & (df.lead_time == lead_time),['date']+vmax_vals]
    tmp.plot(x='date',title='Storm '+str(storm_id)
            +'\nVMAX and estimates for lead time '+str(lead_time))


#tmp = storms.reset_index()
##tmp[tmp.lead_time==0].value_counts()
#summary= tmp.groupby('lead_time')['vmax_miss','vmax_ivcn_miss'
#                    ,'vmax_nhc_miss','vmax_hwrf_miss'].agg(['mean'])
#
##print(storms.vmax_miss.value_counts().sort_index())
#print(storms.vmax_nhc_miss.value_counts().sort_index())


## PERFORMANCE BY LEAD TIME
#sep_models = pd.read_csv(wk_dir+'1_model_preds_separate.csv',index_col=0)
#df_c  = pd.read_csv(wk_dir+'0_clean_data.csv',index_col=0)
#df_c = df_c[(df_c.vmax != -9999) & (df_c.vmax_ivcn != -9999) & (df_c.vmax_nhc != -9999) & df_c.lead_time > 0]
#
#cols = []
#competitors = ['nhc']
#for competitor in competitors:
#    df_c[competitor] = np.abs(df_c['vmax_'+competitor] - df_c.vmax)
#    cols = cols+[competitor]
#
#model_perf = df_c.groupby(['lead_time'])[cols].agg('mean')
#model_perf=model_perf.merge(sep_models,how='left',left_index=True,
#                            right_index=True)
#model_perf.columns = competitors + ['nn_separate']
#fig = plt.figure()
#ax1 = fig.add_subplot(111)
#model_perf.plot(ax=ax1,ylim=0
#          ,xticks=[6*(x+.5) for x in range(16)]
#          ,title='Model MAE by lead time',figsize=(9,7))

#model_perf = df_nm.groupby(['lead_time','dataset'])[cols].agg('mean').reset_index()
#fig = plt.figure()
#ax1 = fig.add_subplot(221)
#ax2 = fig.add_subplot(222)
#model_perf[model_perf.dataset == 'atlantic'].plot(x='lead_time',ax=ax1
#          ,xticks=[3,6,9,12,15,18,21,24]
#          ,title='Atlantic',sharey=True,figsize=(11,8))
#model_perf[model_perf.dataset == 'east_pacific'].plot(x='lead_time',ax=ax2
#          ,xticks=[3,6,9,12,15,18,21,24],sharey=True,title='East Pacific')
#print(model_perf.sort_values('dataset').to_clipboard())