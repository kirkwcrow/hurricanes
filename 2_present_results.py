import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import t

### PROGRAM PARAMETERS ###
wk_dir     = "D:\\System\\Documents\\ACADEMIC\\HF\\Data\\2018_12_21\\"
file_perf  = '1_model_performance.csv'
file_gars  = '1_garson_importance.csv'
file_boot  = '1_bootstrap_results.csv'
file_seq   = '1_seq_predictions.csv'
pred_names = 'pred_names.csv'

### MODEL PARAMETERS ###
gar_preds   = ['vmax_hwrf','vmax_op_t0','V1'] #predictors to plot

pretty_models = {'nhc_mae':'NHC','val_mae':'Neural net','hwrf_mae':'HWRF'}

#%%
lead_times  = [3*(x+1) for x in range(16)]

### EXECUTE ###
var_names = pd.read_csv(wk_dir+pred_names,index_col=0).to_dict()['Name']

results = pd.read_csv(wk_dir+file_perf)
ax1 = results.rename(columns=pretty_models).plot(x=['lead_time']
            ,y=list(pretty_models.values())
            ,figsize=(6.5,5)
            ,ylim=0)
ax1.set_ylabel('Mean absolute error (knots)')
ax1.set_xlabel('Lead time (hours)')

# GARSON VARIABLE IMPORTANCE
gar=pd.read_csv(wk_dir+file_gars,index_col=0)
gar['max']=gar.max(axis=1)
gar.sort_values(by='max',ascending=False,inplace=True)
del gar['max']
gar=gar.transpose()
to_plot = gar.iloc[:,0:5].copy()
to_plot.rename(columns = var_names,inplace=True)
ax=to_plot.plot(ylim=0,figsize=(6.5,5))
ax.set_ylabel('Garson variable importance')
ax.set_xlabel('Lead time (hours)')

gar_out = gar.transpose()

#%%

# SEQUENTIAL TEST RAPID INTENSIFICATION DETECTION
threshold = 30
seq = pd.read_csv(wk_dir+file_seq,index_col=0)

# get vmax and vmax_op at t=0
orig = pd.read_csv(wk_dir+'0_clean_data.csv',
                   usecols=['storm_id','lead_time','date','vmax','vmax_op'])
orig = orig[orig.lead_time == 0]
del orig['lead_time']
seq=seq.merge(orig,on=['storm_id','date'],how='left',suffixes=('','_t0'))

gil = seq[seq.lead_time == 24].copy()
gil['true_ri'] = (gil.vmax - gil.vmax_t0 > threshold).astype(int)
model_skill = []
models = ['nhc','hwrf','pred_seq','hwfi']
for m in models:
    pred = m+'_ri' # for convenient reference
    gil[pred] = (gil['vmax_'+m]-gil.vmax_op_t0 > threshold).astype(int)
    
    # components of Gilbert skill score
    hits = ((gil[pred] == 1) & (gil['true_ri'] == 1)).sum()
    miss = ((gil[pred] == 0) & (gil['true_ri'] == 1)).sum()
    f_a  = ((gil[pred] == 1) & (gil['true_ri'] == 0)).sum()
    total = len(gil)
    hits_random = (hits+miss)*(hits+f_a)/total
    print(hits_random)
    
    # compute score
    skill = (hits-hits_random)/(hits+miss+f_a-hits_random)
    model_skill.append((hits,miss,f_a,skill))

gil_res = pd.DataFrame.from_records(model_skill,index=models
                                    ,columns=['Hits',
                                              'Misses',
                                              'False alarms',
                                              'Gilbert skill score'])

gil.to_csv(wk_dir+'2017_storm_preds_with_ri_detection.csv',index=False)

#%%
# SEQUENTIAL TEST RESULTS
seq = pd.read_csv(wk_dir+file_seq,index_col=0)

def seq_results(df1,err_type='MAE',plot=True):
    df = df1.copy()
    for var in ['nhc','pred_seq','hwrf','hwfi']:
        df[var+'_diff']=np.abs(df['vmax_'+var]-df.vmax)
        if err_type == 'MSE': df[var+'_diff']=df[var+'_diff']**2

    perf_leadtime = df.groupby('lead_time').agg('mean').iloc[:,-4:] ## CAREFUL HERE: indexing by position
    perf_leadtime.columns = ['NHC','Neural net','HWRF','HWFI']
    if plot: 
        ax=perf_leadtime.plot(xlim=[df.lead_time.min()-1,df.lead_time.max()+1])
            #'2017 out of sample performance by lead time ('+err_type+')')
        y_lab = 'Mean absolute error (knots)'
        if err_type == 'MSE': y_lab = 'Mean squared error (knots)'
        ax.set_ylabel(y_lab)
        ax.set_xlabel('Lead time (hours)')

    return perf_leadtime

def seq_res_basin(df,err_type='MAE'):
    atl = seq_results(df[df.dataset == 'atlantic'].copy(),err_type,False)
    pac = seq_results(df[df.dataset == 'east_pacific'],err_type,False)
    fig, axes = plt.subplots(nrows=1, ncols=2)
    atl.plot(ax=axes[0],title='Atlantic',sharey=True,figsize=(9,5))
    pac.plot(ax=axes[1],title='Pacific',sharey=True)

    fig.suptitle('Model performance by basin ('+err_type+')')

seq_results(seq,'MSE')
seq_res_basin(seq)
seq_res_basin(seq,'MSE')

# BOOTSTRAP SEQUENTIAL
#seq = pd.read_csv(wk_dir+file_seq,index_col=0)

def bsample(df,by_id='none'):
    if by_id == 'none':
        return df.iloc[np.random.randint(0,len(df),size=len(df))]    
    else:
        groups=pd.DataFrame(data=df[by_id].unique(),columns=[by_id])
        bootstr=groups.loc[np.random.randint(0,len(groups),size=len(groups))]
        return df.merge(bootstr,'right',by_id)

for pred_var in ['pred_seq','hwrf']:
    seq['abs_err_'+pred_var] = np.abs(seq.vmax-seq['vmax_'+pred_var])

def bootstrap_ci(df,k,quantiles):
    for i in range(k):
        v_keep = ['abs_err_hwrf','abs_err_pred_seq','lead_time']
        tmp=bsample(seq,'storm_id')[v_keep].groupby(
                'lead_time').agg('mean')
        tmp['diff'] = tmp['abs_err_pred_seq'] - tmp['abs_err_hwrf']
        if i == 0: mae = tmp
        else: mae=mae.join(tmp['diff'],rsuffix=str(i))
    ci = mae.quantile(quantiles,axis=1).transpose()
    ci['margin'] = (ci.iloc[:,1]-ci.iloc[:,0])/2
    print(ci.head())
    return ci

ci=bootstrap_ci(seq,100,[0.05,0.95])
a=seq_results(seq)
plt.errorbar(ci.index,y=ci[0.05]+ci['margin'],yerr=ci.margin,fmt='none',elinewidth=1,ecolor='black',capsize=3)
plt.axhline(y=0,linewidth=1, color='gray',ls='--')

#%% 
# SHOW PREDICTIONS
dv_dir     = "D:\\System\\Documents\\ACADEMIC\\HF\\"
core_vars= ['storm_id','lead_time','date','vmax','vmax_nhc','vmax_pred_seq'
              ,'vmax_hwrf']
seq = pd.read_csv(wk_dir+file_seq,index_col=0,usecols=core_vars)
seq.date = pd.to_datetime(seq.date,yearfirst=True)

seq=seq[seq.lead_time == 24]
start_t = seq.groupby('storm_id').agg({'date':'min'})
seq = seq.join(start_t,rsuffix='_st')
seq['t'] = (seq['date'] - seq['date_st']).apply(lambda x: x.total_seconds()/3600)

for s_id in list(seq.index.unique()):
    tmp = seq.loc[[s_id],['t']+core_vars[-4:]]
    if len(tmp) > 10: 
        ax=tmp.plot(x='t',title='24-hr predictions versus best-track VMAX'
                    +'\n(storm '+str(s_id)+')')
        ax.set_ylabel('Windspeed (knots)')
        ax.set_xlabel('Time since '+str(start_t.at[s_id,'date'])+' (hours)')
        plt.savefig(dv_dir+'Plots\\OOS_prediction_plots\\'+str(s_id)+'.png')

#%%
# BOOTSTRAP CROSS-VALIDATION

results_out = results.copy()
boot_raw = pd.read_csv(wk_dir+file_boot)
n = len(boot_raw)
alpha = 0.05
mae_estimate = results['val_mae'] #np.mean(boot_raw)
range = (np.percentile(boot_raw,100*(1-alpha/2),axis=0)
        -np.percentile(boot_raw,100*alpha/2,axis=0))
ci_lower     = mae_estimate-range/2
ci_upper     = mae_estimate+range/2

results['bootstr_mean'] = list(mae_estimate)
results['bootstr_ci_lb'] = list(ci_lower)
results['bootstr_ci_ub'] = list(ci_upper)

results.plot(x=['lead_time']
            ,y=['nhc_mae','hwrf_mae','bootstr_mean','bootstr_ci_lb','bootstr_ci_ub']
            ,ylim=0
            ,figsize=(8,5.5)
            ,color=['C0','C1','k','C7','C7']
            ,style=['-']*3+['--']*2
            ,title='Model MAE by lead time\nwith bootstrapped NN performance, '+
                str(100*(1-alpha))+'% CI (n=100 per lead time)')

results_out['bootstrap_std'] = list(np.std(boot_raw))