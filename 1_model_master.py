import numpy as np
import pandas as pd
from scipy.stats import t
from sklearn.linear_model import LinearRegression
from keras.models import Model,Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers import Input,Dense,Concatenate,Dropout
import datetime

### PROGRAM PARAMETERS ###
wk_dir     = "D:\\System\\Documents\\ACADEMIC\\HF\\Data\\2017_10_25\\"
filename   = '0_clean_data.csv'
plot_subti = 'shallow separate'
print_work = 0
rand_seed  = 46
n_parts    = 10

### MODEL PARAMETERS ###
ft_ready    = ['dataset_ind','V6_x','V6_y','V8_x','V8_y','V6_y_miss','V8_y_miss'] # no processing
ft_to_norm  = ['vmax_op_t0']#,'n_miss']#,'vmax_pred_prev'] # normalize only
ft_to_imp   = ['vmax_hwrf'] #,'vmax_op_t0'] # impute and normalize
fit_resids  = 0
init_vals   = 0
miss_ind    = 0
impute      = 0
lead_t_ind  = 0
lead_times  = [3*(x+1) for x in range(16)]
epochs      = 40
batch_size  = 30
cost_fn     = 'mean_squared_error'
competitor  = 'nhc'
response    = 'vmax'

### FUNCTIONS ###
def create_model(p):
    model = Sequential()
    model.add(Dense(200, input_dim=p, kernel_initializer='normal'
                    , activation='sigmoid'))
    model.add(Dense(30, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, kernel_initializer='normal',activation='linear'))
    model.compile(loss=cost_fn, optimizer='adam')
    return model

def create_bn_model(p):
    d_pct = 0.3
    l_size = 250
    model = Sequential()
    model.add(Dense(l_size, input_dim=p, kernel_initializer='normal'
                    , activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(d_pct))
    model.add(Dense(l_size, kernel_initializer='normal',activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(d_pct))
    model.add(Dense(l_size, kernel_initializer='normal',activation='relu'))
    model.add(Dropout(d_pct))
    model.add(Dense(1, kernel_initializer='normal',activation='linear'))
    model.compile(loss=cost_fn, optimizer='adam')
    return model

def peek(df,varlist=[],nrows=5): # just a quick way to look at the dataframe
    defaults = ['storm_id','date','lead_time','vmax','vmax_hwrf']
    print(df[defaults+varlist].head(nrows))

def s_print(str1,str2,total_length=75):
    if len(str1)+len(str2) <= 75:
        print(str1+'.'*(total_length - len(str1+str2))+str2)
    else:
        pt1=(str1+str2)[:total_length]
        pt2=(str1+str2)[total_length:]
        print(pt1+'\n'+(total_length-len(pt2))*' '+pt2)

def print_settings():
    print('\n\n'+str(datetime.datetime.now()))
    s_print('General:','n='+str(len(hf))+', storms='+str(len(hf.storm_id.unique()))
          +', p='+str(p)+', seed='+str(rand_seed)+', epochs='+str(epochs)
          +', batch size='+str(batch_size))
    s_print('Cost function:',cost_fn)
    s_print('Additional features:',str(ft_ready+ft_to_norm+ft_to_imp))
    s_print('Predict HWRF residuals: ',str(fit_resids))
    s_print('Initial values:', str(init_vals))
    s_print('Missing indicators:',str(miss_ind))
    s_print('Imputation:',str(impute))
    s_print('Lead times:',str(lead_times))
    
def normalize_features(df,feature_list,test_pt=-1):
    if test_pt != -1:
        training_obs = df.partition != test_pt
    else:
        training_obs = pd.Series(np.ones(len(df)))
    for ft in feature_list:
         train_data  = df.loc[(df[ft]!=-9999) & training_obs,ft]
         df[ft+'_n'] = ( ((df[ft]-train_data.mean())/train_data.std())*
                        (df[ft] != -9999).astype(int) ) # norm. non-missing

def impute_features(df,feature_list,features_norm_only,test_pt):
    tmp = df.copy()
    for ft in feature_list: # replace with means
        mean=tmp.loc[(tmp[ft+'_miss']==0)&(tmp.partition != test_pt),ft].mean()
        tmp.loc[tmp[ft+'_miss']==1,ft] = mean
    
    for imp_pass in ['','_imp']: # fits on meaned vals first, then imputed
        for ft in feature_list: # regress ft on cov
            cov = [x+imp_pass for x in list(feature_list) if x != ft]
            non_miss = tmp[tmp[ft+'_miss'] ==0]
                   
            reg = LinearRegression()
            reg.fit(non_miss[cov].values,non_miss[ft].values)
            tmp[ft+'_imp'] = (reg.predict(tmp[cov].values)*tmp[ft+'_miss']
                            +df[ft]*(1-tmp[ft+'_miss']) )
    all_features = [a+'_imp' for a in feature_list] + features_norm_only
    for ft in all_features:
        train_data = tmp.loc[tmp.partition != test_pt,ft]
        tmp[ft] = (tmp[ft] - train_data.mean())/train_data.std()
        df[ft+'_n'] =  tmp[ft]

def kfold_partition(df,grouping_var,k):
    groups=pd.DataFrame(data=df[grouping_var].unique(),columns=[grouping_var])
    groups['partition']=pd.qcut(
            np.random.rand(len(groups)),k,labels=range(k)).codes
    return df.merge(groups,how='left',on=grouping_var)

def fit_SLR_hwrf(df,response,test_pt):
    reg = LinearRegression()
    reg.fit(df.loc[df.partition != test_pt,'vmax_hwrf'].values.reshape(-1,1),
            df.loc[df.partition != test_pt,response].values)
    return reg.predict(df['vmax_hwrf'].values.reshape(-1,1))

def apply_NN_1pt(df,model,test_pt,features,response,e,b_size): 
    df_X1 = df.loc[df.partition!=test_pt,features]
    df_Y1 = df.loc[df.partition!=test_pt,response]

    df_X2 = df.loc[df.partition==test_pt,features]
    df_Y2 = df.loc[df.partition==test_pt,response]

    model.load_weights(wk_dir+'nn_initial_weights.h5')
    model.fit(df_X1.values,df_Y1.values,epochs=e,batch_size=b_size
              ,validation_data=(df_X2.values,df_Y2.values)
              ,verbose=print_work)
    return model.predict(df_X2.values,batch_size=b_size)

def apply_model(df,model,ft_imp,ft_norm,ft__ready,response,e,b_size):
    ft_to_imp=cleanse(df,ft_imp)
    ft_to_norm=cleanse(df,ft_norm)
    ft_ready=cleanse(df,ft__ready)
    pts = df.partition.unique()
    pts.sort()
    df[response+'_pred'] = np.zeros(len(df))
    for test_pt in pts:
        print(str(test_pt),end='')
        df_n = df.copy(deep=True)
        
        if fit_resids == 1:
            df_n[response+'_slr'] = fit_SLR_hwrf(df_n,response,test_pt)
            df_n['hwrf_resid'] = df_n[response] - df_n[response+'_slr']
            nn_response = 'hwrf_resid'
        else: nn_response = response
            
        # ------------- prepare features ---------------------------------#
        if impute == 1: impute_features(df_n,ft_to_imp,ft_to_norm,test_pt)
        else: normalize_features(df_n,ft_to_imp+ft_to_norm,test_pt)
        features = ( [x+'_n' for x in ft_to_norm] 
                    +[x+impute*'_imp'+'_n' for x in ft_to_imp] 
                    + ft_ready )
        if miss_ind==1: features = features + [x + '_miss' for x in ft_to_imp]
        
        nn_preds=apply_NN_1pt(df_n,model,test_pt,features,nn_response,e,b_size)
        
        if fit_resids == 1:
            test_hwrf_vals = df_n.loc[df.partition==test_pt
                                      ,response+'_slr'].values
            preds = [sum(x) for x in zip(nn_preds,test_hwrf_vals)]
        else: preds = nn_preds
        
        df.loc[df.partition == test_pt,response+'_pred'] = preds
    print('')
    return np.abs(df[response+'_pred'] - df[response]).mean() # MAE

def bsample(df,by_id):
    groups=pd.DataFrame(data=df[by_id].unique(),columns=[by_id])
    bootstr=groups.loc[np.random.randint(0,len(groups),size=len(groups))]
    return df.merge(bootstr,'right',by_id)

def bootstrap(df,n):
    res = []
    for i in range(n):
        df_bs = bsample(df,'storm_id')
        df_bs=kfold_partition(df_bs,'storm_id',n_parts) 
        mae = apply_model(df_bs,nn_model,ft_to_impute,ft_to_norm,ft_ready,
                      response,epochs,batch_size)
        res.append(mae)
    mae_est = np.mean(res)
    ci = t.interval(0.95,n-1,mae_est,np.std(res))
    
    print_settings()
    print('\nBootstrap results (n='+str(n)+'):\n'
          +'Mean: '+str(mae_est)
          +'\n95% CI: '+str(ci))
    return res

def single_run(df):
    df = kfold_partition(df,'storm_id',n_parts)
    apply_model(df,nn_model,ft_to_impute,ft_to_norm,ft_ready,
                response,epochs,batch_size)
    print_settings()
    return df

#  compares results of model upon perturbation of a feature 
def perturbed_runs(df,feature,std_list,competitor=''):
    res = []
    for std in std_list:
        df_p = df.copy()
        df_p['vmax_t0'] = df_p['vmax_t0']+np.random.randn(len(df_p))*std
        df_p=kfold_partition(df_p,'storm_id',n_parts) 
        mae = apply_model(df_p,nn_model,ft_to_impute,ft_to_norm,ft_ready,
                        response,epochs,batch_size)        
        res.append((std,mae))
    results = pd.DataFrame(data=res,columns=['perturbation','nn_mae'])
    
    if competitor != '':
        comp_mae = (np.abs(df['vmax_'+competitor]-df['vmax'])).mean()
        results[competitor+'_mae'] = comp_mae
    results.plot(x='perturbation',
                 title='NN model MAE by perturbation of '+feature)
    return results

def cleanse(df,ft_list): # delete features with 1 unique value
    ft_clean = []
    for ft in ft_list:
        if len(df[ft].unique())>1:
            ft_clean.append(ft)
        else:
            print('Note: '+ft+' removed (no unique values).')
    return ft_clean

def get_next_dataframe(df_orig,df_preds,lead_time): # for iterated predictions
    tmp = df_orig[df_orig['lead_time'] == lead_time].copy()
    if lead_time == 3:
        tmp['vmax_pred_prev'] = tmp['vmax_op_t0']
    else:
        past_pred = df_preds.loc[df_preds['lead_time']==lead_time-3
               ,['storm_id','date','vmax_pred']].copy()
        past_pred.rename(columns={'vmax_pred':'vmax_pred_prev'},inplace=True)
        tmp = tmp.merge(past_pred,how='left',on=['storm_id','date'])
        tmp.loc[tmp.vmax_pred_prev.isnull(),'vmax_pred_prev']=-9999 # WHY DO MISSING VALUES HAPPEN
    return tmp

def multi_run(df,lead_times):
    df_all = kfold_partition(df,'storm_id',n_parts)
    df_preds = pd.DataFrame()
    for lt in lead_times:
        print('Lead time: '+str(lt))
        tmp = get_next_dataframe(df_all,df_preds,lt)
        apply_model(tmp,nn_model,ft_to_impute,ft_to_norm,ft_ready,
                        response,epochs,batch_size)
        df_preds = df_preds.append(tmp)
    return df_preds

def all_results(df,competitor):
    df['abs_err_'+competitor] = np.abs(df['vmax_'+competitor]-df[response])
    df['abs_err_hwrf']        = np.abs(df['vmax_hwrf']     -df[response])
    df['abs_err_pred']        = np.abs(df[response+'_pred']-df[response])
    res = []
    parts = list(range(df.partition.max()+1))+['all']
    lead_times = df.lead_time.unique()
    for lt in lead_times:
        for pt in parts:
            if pt == 'all': rows = (df.partition > -1) & (df.lead_time == lt)
            else:           rows = (df.partition == pt) & (df.lead_time == lt)
            n_obs = rows.sum()
            test_storms = len(df[rows]['storm_id'].unique())
            val_mae  = df.loc[rows,'abs_err_pred'].mean()
            hwrf_mae = df.loc[rows,'abs_err_hwrf'].mean()
            comp_mae = df.loc[rows,'abs_err_'+competitor].mean()
            res.append((n_obs,lt,test_storms,val_mae,hwrf_mae
                        ,comp_mae,val_mae-comp_mae,
                        (val_mae-comp_mae)/comp_mae))
    
    result = pd.DataFrame(res,index=parts*len(lead_times)
                          ,columns=['n','lead_time','n_storms','val_mae'
                                    ,'hwrf_mae',
                                    competitor+'_mae','difference','pct_diff'])
    return result

def sum_results(df,competitor,plot_sub=''):
    all_res = all_results(df,competitor)
    sum_res = all_res[all_res.index=='all']
    print(sum_res)
    if len(df.lead_time.unique()) > 1:
        sum_res.plot(x=['lead_time'],y=[competitor+'_mae','val_mae','hwrf_mae']
            ,ylim=0,title='Model MAE by lead time\n'+plot_sub)
    return sum_res

### PREP ###
np.random.seed(rand_seed)
hf_raw = pd.read_csv(wk_dir+filename,index_col=0,parse_dates=['date']) 
hf=hf_raw[(hf_raw.vmax != -9999) & (hf_raw['vmax_'+competitor] != -9999) 
         &(hf_raw['vmax_op_t0'] != -9999)]
hf=hf[hf.lead_time.isin(lead_times)]

if lead_t_ind and len(lead_times) > 1:
    for lt in lead_times:
        ft_ready = ft_ready + ['lead_time_'+str(lt)]

last_var = 'V62'
base_vars = list(hf.loc[1:2,'V1':last_var])
ft_to_impute = base_vars+ ft_to_imp + (
        init_vals*list(hf.loc[1:2,'V1_t0':last_var+'_t0']) )
p = (1+miss_ind)*len(ft_to_impute)+len(ft_to_norm)+len(ft_ready)
nn_model = create_model(p)
nn_model.save_weights(wk_dir+'nn_initial_weights.h5')

### EXECUTE ###

#bootstrap_results = bootstrap(3)
#hf = single_run(hf)

hf = multi_run(hf,lead_times)
print_settings()
res=sum_results(hf,competitor,plot_subti)

#res.to_csv(path_or_buf=wk_dir+'1_model_preds_combined.csv',index=False)
#hf.to_csv(path_or_buf=wk_dir+'1_model_preds.csv',index=True) 