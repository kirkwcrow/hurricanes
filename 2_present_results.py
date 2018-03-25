import numpy as np
import pandas as pd
from scipy.stats import t

### PROGRAM PARAMETERS ###
wk_dir     = "D:\\System\\Documents\\ACADEMIC\\HF\\Data\\2018_03_17\\"
file_perf  = '1_model_performance.csv'
file_gars  = '1_garson_importance.csv'
file_boot  = '1_bootstrap_results.csv'
file_seq   = '1_seq_predictions.csv'
pred_names = 'pred_names.csv'

### MODEL PARAMETERS ###
ft_ready    = ['V6_x','V6_y','V8_x','V8_y','dataset_ind'] #'V6_y_miss','V8_y_miss'  #  no processing
ft_to_norm  = ['vmax_op_t0']#,'vmax_pred_prev'] # normalize only
ft_to_imp   = ['vmax_hwrf'] #,'vmax_op_t0'] # impute and normalize

gar_preds   = ['vmax_hwrf','vmax_op_t0','V1'] #predictors to plot

pretty_models = {'nhc_mae':'NHC','val_mae':'Neural net','hwrf_mae':'HWRF'}

#lead_times  = [3*(x+1) for x in range(16)]

### EXECUTE ###
#var_names = pd.read_csv(wk_dir+pred_names,index_col=0).to_dict()['Name']
#
#results = pd.read_csv(wk_dir+file_perf)
#ax1 = results.rename(columns=pretty_models).plot(x=['lead_time']
#            ,y=list(pretty_models.values())
#            ,figsize=(6.5,5)
#            ,ylim=0)
#ax1.set_ylabel('Mean absolute error (knots)')
#ax1.set_xlabel('Lead time (hours)')
#
##garson variable importance
#gar=pd.read_csv(wk_dir+file_gars,index_col=0)
#gar['max']=gar.max(axis=1)
#gar.sort_values(by='max',ascending=False,inplace=True)
#del gar['max']
#gar=gar.transpose()
#to_plot = gar.iloc[:,0:5].copy()
#to_plot.rename(columns = var_names,inplace=True)
#ax=to_plot.plot(ylim=0,figsize=(6.5,5))
#ax.set_ylabel('Garson variable importance')
#ax.set_xlabel('Lead time (hours)')
#
#gar_out = gar.transpose()

# sequential tests
seq = pd.read_csv(wk_dir+file_seq,index_col=0)

def seq_results(df,err_type='MAE'):
    for var in ['nhc','pred_seq','hwrf']:
        df[var+'_diff']=np.abs(df['vmax_'+var]-df.vmax)
        if err_type == 'MSE': df[var+'_diff']=df[var+'_diff']**2
    perf_leadtime = seq.groupby('lead_time').agg('mean').iloc[:,-3:]
    perf_leadtime.columns = ['NHC','Neural net','HWRF']
    perf_leadtime.plot(title='2017 out of sample performance by lead time ('+
                            err_type+')')
    return perf_leadtime

seq_results(seq)
seq_results(seq,'MSE')

# bootstrap results


#results_out = results.copy()
#boot_raw = pd.read_csv(wk_dir+file_boot)
#n = len(boot_raw)
#alpha = 0.05
#mae_estimate = results['val_mae'] #np.mean(boot_raw)
#range = (np.percentile(boot_raw,100*(1-alpha/2),axis=0)
#        -np.percentile(boot_raw,100*alpha/2,axis=0))
#ci_lower     = mae_estimate-range/2
#ci_upper     = mae_estimate+range/2
#
#results['bootstr_mean'] = list(mae_estimate)
#results['bootstr_ci_lb'] = list(ci_lower)
#results['bootstr_ci_ub'] = list(ci_upper)
#
#results.plot(x=['lead_time']
#            ,y=['nhc_mae','hwrf_mae','bootstr_mean','bootstr_ci_lb','bootstr_ci_ub']
#            ,ylim=0
#            ,figsize=(8,5.5)
#            ,color=['C0','C1','k','C7','C7']
#            ,style=['-']*3+['--']*2
#            ,title='Model MAE by lead time\nwith bootstrapped NN performance, '+
#                str(100*(1-alpha))+'% CI (n=100 per lead time)')
#
#results_out['bootstrap_std'] = list(np.std(boot_raw))