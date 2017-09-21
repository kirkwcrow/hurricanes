import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
#from keras.models import Model
#from keras.layers import *
from keras.models import Sequential
from keras.layers import Dense
import datetime

### PROGRAM PARAMETERS ###
wk_dir     = "D:\\System\\Documents\\ACADEMIC\\HF\\Data\\"
filename   = '0_clean_data.csv'
print_work = 1
rand_seed  = 46

### MODEL PARAMETERS ###
ft_ready    = ['dataset_ind'] # no processing
ft_to_norm  = [] #['lead_time'] # normalize only
ft_to_imp   = ['vmax_hwrf','vmax_t0'] # impute and normalize
fit_resids  = 1
init_vals   = 0
miss_ind    = 1
impute      = 1
lead_times  = [3,6,9,12,15,18,21,24]
epochs      = 20
batch_size  = 15
cost_fn     = 'mean_squared_error'
competitor  = 'ivcn'
response    = 'vmax'

## FUNCTIONS ###
def create_model(p):
    model = Sequential()
#    model.add(Dropout(0.2, input_shape=(p,))) # % of features dropped
    model.add(Dense(1000, input_dim=p, kernel_initializer='normal'
                    , activation='sigmoid'))
#    model.add(Dense(300, kernel_initializer='normal', activation='relu'))
    model.add(Dense(30, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, kernel_initializer='normal',activation='linear'))
    model.compile(loss=cost_fn, optimizer='adam')
    return model

#def create_model(p_essential,p_droppable):
#    essential_ft = Input((p_essential,))
#    droppable_ft = Input((p_droppable,))
#
#    dropped = Dropout(0.2)(droppable_ft) # % of features dropped
#    completeInput = Concatenate()([essential_ft,droppable_ft])        
#
#    layer_1 = Dense(1000,kernel_initializer='normal'
#                    ,activation='sigmoid')(completeInput)
#    layer_2 = Dense(30, kernel_initializer='normal', activation='relu')(layer_1)
#    output = Dense(1, kernel_initializer='normal',activation='linear')(layer_2)
#
#    model = Model([essential_ft,droppable_ft],output)
#    model.compile(loss=cost_fn, optimizer='adam')
#    return model


def s_print(str1,str2,total_length=70):
    print(str1+'.'*(total_length - len(str1+str2))+str2)


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
    
    
def normalize_features(df,feature_list,test_pt):
    for ft in feature_list:
         train_data  = df.loc[(df[ft]!=-9999) & (df.partition != test_pt),ft]
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


def apply_model(df,model,ft_to_imp,ft_to_norm,ft_ready,response,e,b_size):
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
    return np.abs(df[response+'_pred'] - df[response]).mean() # MAE

def bsample(df,by_id):
    return df.copy()

def sum_results(df,competitor):
    df['abs_err_'+competitor] = np.abs(df['vmax_'+competitor]-df[response])
    df['abs_err_pred']        = np.abs(df[response+'_pred']-df[response])
    res = []
    parts = list(range(df.partition.max()+1))+['all']
    for pt in parts:
        if pt == 'all': rows = (df.partition > -1)
        else:           rows = (df.partition == pt)
        n_obs = rows.sum()
        test_storms = len(df[rows]['storm_id'].unique())
        val_mae  = df.loc[rows,'abs_err_pred'].mean()
        comp_mae = df.loc[rows,'abs_err_'+competitor].mean()
        res.append((n_obs,test_storms,val_mae,comp_mae,val_mae-comp_mae,
                    (val_mae-comp_mae)/comp_mae))
    result = pd.DataFrame(res,index=parts,columns=['n','n_storms','val_mae',
                          competitor+'_mae','difference','pct_diff'])
    print('\n'+str(result))
    return result


### EXECUTE ###
np.random.seed(rand_seed)
hf_raw = pd.read_csv(wk_dir+filename,index_col=0,parse_dates=['date']) 
hf=hf_raw[(hf_raw.vmax != -9999) & (hf_raw['vmax_'+competitor] != -9999)]
hf=hf[hf.lead_time.isin(lead_times)]
base_vars = list(hf.loc[1:2,'V1':'V62'])
ft_to_impute = base_vars+ ft_to_imp + (
        init_vals*list(hf.loc[1:2,'V1_t0':'V62_t0']) )
p = (1+miss_ind)*len(ft_to_impute)+len(ft_to_norm)+len(ft_ready)
nn_model = create_model(p)
nn_model.save_weights(wk_dir+'nn_initial_weights.h5')

results = []
for lt in lead_times:
    print('Lead time: '+str(lt))
    tmp = hf[hf.lead_time == lt]
    tmp=kfold_partition(tmp,'storm_id',10) 
    mae = apply_model(tmp,nn_model,ft_to_impute,ft_to_norm,ft_ready,
                    response,epochs,batch_size)
    results.append((lt,mae))
print_settings()
sum_results(hf,competitor)
print(results)

res_df =pd.DataFrame(data=results,columns=['lead_time','pred_sep_mae'])
res_df.to_csv(path_or_buf=wk_dir+'1_model_preds_separate.csv',index=False)
#hf.to_csv(path_or_buf=wk_dir+'1_model_preds.csv',index=True)
