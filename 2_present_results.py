import numpy as np
import pandas as pd
from scipy.stats import t

### PROGRAM PARAMETERS ###
wk_dir     = "D:\\System\\Documents\\ACADEMIC\\HF\\Data\\2017_10_25\\"
file_perf  = '1_model_performance.csv'
file_gars  = '1_garson_importance.csv'
file_boot  = '1_bootstrap_results.csv'

### MODEL PARAMETERS ###
ft_ready    = ['V6_x','V6_y','V8_x','V8_y','dataset_ind'] #'V6_y_miss','V8_y_miss'  #  no processing
ft_to_norm  = ['vmax_op_t0']#,'vmax_pred_prev'] # normalize only
ft_to_imp   = ['vmax_hwrf'] #,'vmax_op_t0'] # impute and normalize

gar_preds   = ['vmax_hwrf','vmax_op_t0','V1'] #predictors to plot

lead_times  = [3*(x+1) for x in range(16)]

### EXECUTE ###
results = pd.read_csv(wk_dir+file_perf)
results_out = results.copy()

# bootstrap results
boot_raw = pd.read_csv(wk_dir+file_boot)
n = len(boot_raw)
alpha = 0.01
mae_estimate = np.mean(boot_raw)
std_estimate = np.std(boot_raw)
standard_err = std_estimate/np.sqrt(n)
ci_lower     = mae_estimate+std_estimate*t.ppf(alpha/2,n-1)/np.sqrt(n)
ci_upper     = mae_estimate+std_estimate*t.ppf(1-alpha/2,n-1)/np.sqrt(n)
#ci = t.interval(0.95,n-1,mae_estimate,np.std(boot_raw))

results['bootstr_mean'] = list(mae_estimate)
results['bootstr_ci_lb'] = list(ci_lower)
results['bootstr_ci_ub'] = list(ci_upper)

results.plot(x=['lead_time']
            ,y=['nhc_mae','hwrf_mae','bootstr_mean','bootstr_ci_lb','bootstr_ci_ub']
            ,ylim=0
            ,figsize=(10,7)
            ,color=['C0','C1','k','C7','C7']
            ,style=['-']*3+['--']*2
            ,title='Model MAE by lead time\nwith bootstrapped NN performance, 99% CI (n=100 per lead time)')

results_out['bootstrap_se'] = list(standard_err)


# garson variable importance
#gar=pd.read_csv(wk_dir+file_gars,index_col=0)
#gar['max']=gar.max(axis=1)
#gar.sort_values(by='max',ascending=False,inplace=True)
#del gar['max']
#gar=gar.transpose()
#ax= gar.iloc[:,:5].plot(ylim=0
#    ,title='Garson variable importance by lead time'+
#                            '\n(top five predictors)'
#    ,figsize=(8,6))
#ax.set_ylabel('Garson variable importance')
#ax.set_xlabel('lead time')
#
#gar_out = gar.transpose()
#gar_out['max']