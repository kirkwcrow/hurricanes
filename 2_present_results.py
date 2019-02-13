import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import t

### PROGRAM PARAMETERS ###
wk_dir     = "D:\\System\\Documents\\ACADEMIC\\HF\\Data\\2018_12_21\\"
file_perf  = '1_model_performance.csv'
file_gars  = '1_garson_importance.csv'
file_boot  = '1_bootstrap_results.csv'
file_seq   = '1_seq_predictions_hwfi.csv'
pred_names = 'pred_names.csv'

### MODEL PARAMETERS ###
gar_preds   = ['vmax_hwfi','vmax_op_t0','V1'] # force plot these predictors

pretty_models = {'nhc_mae':'NHC','val_mae':'Neural net','hwrf_mae':'HWRF'}

#%%
lead_times  = [3*(x+1) for x in range(16)]

### EXECUTE ###
var_names = pd.read_csv(wk_dir+pred_names,index_col=0).to_dict()['Name']
#
#results = pd.read_csv(wk_dir+file_perf)
#ax1 = results.rename(columns=pretty_models).plot(x=['lead_time']
#            ,y=list(pretty_models.values())
#            ,figsize=(6.5,5)
#            ,ylim=0)
#ax1.set_ylabel('Mean absolute error (knots)')
#ax1.set_xlabel('Lead time (hours)')

# GARSON VARIABLE IMPORTANCE
gar=pd.read_csv(wk_dir+file_gars,index_col=0)
gar['max']=gar.max(axis=1)
gar.sort_values(by='max',ascending=False,inplace=True)
del gar['max']
gar=gar.transpose()
to_plot = gar.iloc[:,0:5].copy()
to_plot.rename(columns = var_names,inplace=True)
ax=to_plot.plot(ylim=(0,0.35),figsize=(6.5,5))
ax.set_ylabel('Garson variable importance')
ax.set_xlabel('Lead time (hours)')
ax.legend(loc='left')

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
gil['true_ri'] = (gil.vmax - gil.vmax_t0 >= threshold).astype(int)
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
    model_skill.append((total,hits,miss,f_a,skill))

gil_res = pd.DataFrame.from_records(model_skill,index=models
                                    ,columns=['Total',
                                              'Hits',
                                              'Misses',
                                              'False alarms',
                                              'Gilbert skill score'])
    
gil_res.to_csv(wk_dir+'2017_oos_gilbert_skill.csv',index=True)

# to Latex
toprint= np.round(gil_res.loc[['nhc','hwfi','pred_seq'],:],3)
toprint.index=['NHC','HWFI','NN']
toprint.reset_index(inplace=True)
print(toprint.to_latex(index=False).replace('\\toprule','\hline').replace('\\midrule','\hline').replace('\\bottomrule','\hline'))

gil.to_csv(wk_dir+'2017_storm_preds_with_ri_detection.csv',index=False)

#%%
# SEQUENTIAL TEST RESULTS
seq = pd.read_csv(wk_dir+file_seq,index_col=0)

def seq_results(df1,err_type='MAE',pred='pred_seq',plot=True):
    df = df1.copy()
    for var in ['nhc',pred,'hwrf','hwfi']:
        df[var+'_diff']=np.abs(df['vmax_'+var]-df.vmax)
        if err_type == 'MSE': df[var+'_diff']=df[var+'_diff']**2

    perf_leadtime = df.groupby('lead_time').agg('mean').iloc[:,-4:] ## CAREFUL HERE: indexing by position
    perf_leadtime.columns = ['NHC','Neural net','HWRF','HWFI']
    if plot: 
        ax=perf_leadtime.plot(xlim=[df.lead_time.min()-1,df.lead_time.max()+1],
                                    color=('c','C1','limegreen','darkgreen'),
                                    lw=1.8)
            # https://matplotlib.org/users/dflt_style_changes.html
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
#seq_res_basin(seq)
#seq_res_basin(seq,'MSE')

# BOOTSTRAP SEQUENTIAL
#seq = pd.read_csv(wk_dir+file_seq,index_col=0)

def bsample(df,by_id='none'):
    if by_id == 'none':
        return df.iloc[np.random.randint(0,len(df),size=len(df))]    
    else:
        groups=pd.DataFrame(data=df[by_id].unique(),columns=[by_id])
        bootstr=groups.loc[np.random.randint(0,len(groups),size=len(groups))]
        return df.merge(bootstr,'right',by_id)

for pred_var in ['pred_seq','hwfi']:
    seq['abs_err_'+pred_var] = np.abs(seq.vmax-seq['vmax_'+pred_var])

def bootstrap_ci(df,k,quantiles
                 ,pred_err='abs_err_pred_seq'
                 ,comp_err='abs_err_hwrf'):
    for i in range(k):
        v_keep = ['lead_time',pred_err,comp_err]
        tmp=bsample(df,'storm_id')[v_keep].groupby(
                'lead_time').agg('mean')
        tmp['diff'] = tmp[pred_err] - tmp[comp_err]
        if i == 0: mae = tmp
        else: mae=mae.join(tmp['diff'],rsuffix=str(i))
    ci = mae.quantile(quantiles,axis=1).transpose()
    ci['margin'] = (ci.iloc[:,1]-ci.iloc[:,0])/2
    print(ci.head())
    return ci

ci=bootstrap_ci(seq,100,[0.05,0.95],comp_err='abs_err_hwfi')
a=seq_results(seq)
plt.errorbar(ci.index,y=ci[0.05]+ci['margin'],yerr=ci.margin,fmt='none',elinewidth=1,ecolor='black',capsize=3)
plt.axhline(y=0,linewidth=1, color='gray',ls='--')

#%% Bootstrap CI for 
cv = pd.read_csv(wk_dir+'1_model_preds.csv')
ci = bootstrap_ci(cv,100,[0.05,0.95],pred_err='abs_err_pred',comp_err='abs_err_hwfi')
a=seq_results(cv,pred='pred')
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
# Cross validation results table in LaTeX
cv_res= np.round(pd.read_csv(wk_dir+file_perf),3)
cv_res=cv_res.set_index('lead_time').reset_index()
cv_res.columns = ['Lead time','N. Obs.','N. Storms','HWRF','HWFI','NHC','Neural net']
print(cv_res.to_latex(index=False).replace('\\toprule','\hline').replace('\\midrule','\hline').replace('\\bottomrule','\hline'))