import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
import datetime



### PROGRAM PARAMETERS ###
wk_dir     = "D:\\System\\Documents\\ACADEMIC\\HF\\Data\\"
filename   = '0_clean_data.csv'
model_name = 'basic model'
print_work = 1
rand_seed  = 46

### MODEL PARAMETERS ###
ft_ready    = ['dataset_ind'] # no processing
ft_to_norm  = [] #['lead_time'] # normalize only
ft_to_imp   = ['vmax_t0','vmax_hwrf'] # impute and normalize
miss_ind    = 1
impute      = 1
init_vals   = 1
lead_times  = [3] #,6,9,12,15,18,21,24]
epochs      = 20
batch_size  = 20
competitor  = 'ivcn'
response    = 'vmax'

### FUNCTIONS ###
def create_model(p):
    model = Sequential()
    model.add(Dropout(0.2, input_shape=(p,))) # % of features dropped
    model.add(Dense(1000, input_dim=p, kernel_initializer='normal'
                    , activation='sigmoid'))
#    model.add(Dense(300, kernel_initializer='normal', activation='relu'))
    model.add(Dense(30, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, kernel_initializer='normal',activation='linear'))
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model

def s_print(str1,str2,total_length=70):
    print(str1+'.'*(total_length - len(str1+str2))+str2)

def print_settings():
    print('\n\n'+str(datetime.datetime.now()))
    s_print('Model:',model_name+' - n='+str(len(hf))+', storms='+
          str(len(hf.storm_id.unique()))
          +', p='+str(p)+', seed='+str(rand_seed))
    s_print('Additional features:',str(ft_ready+ft_to_norm+ft_to_imp))
    s_print('Initial values:'  , str(init_vals))
    s_print('Missing indicators:',str(miss_ind))
    s_print('Imputation:',str(impute))
    s_print('Lead times:',str(lead_times))
    s_print('Epochs/Batch size:',str(epochs)+ '/'+str(batch_size))
    
def normalize_features(df,feature_list,test_pt):
    for ft in feature_list:
         train_data  = df.loc[(df[ft]!=-9999) & (df.partition != test_pt),ft]
         df[ft+'_n'] = ( ((df[ft]-train_data.mean())/train_data.std())*
                        (df[ft] != -9999).astype(int)  ) # norm. non-missing

def impute_features(df,feature_list,features_norm_only,test_pt):
    tmp = df.copy()
    for ft in feature_list: # replace with means
        mean=tmp.loc[(tmp[ft+'_miss']==0)&(tmp.partition != test_pt),ft].mean()
        tmp.loc[tmp[ft+'_miss']==1,ft] = mean
    
    for imp_pass in ['','_imp']: # fits on meaned vals first, then imputed
        for ft in feature_list: # regress ft
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

def apply_model(df,model,ft_to_imp,ft_to_norm,ft_ready,response,e,b_size):
    pts = df.partition.unique()
    pts.sort()
    mse_results = []
    df[response+'_pred'] = np.zeros(len(df))
    for test_pt in pts:
        print(str(test_pt),end='')
        df_n = df.copy(deep=True)
        
        if impute == 1: impute_features(df_n,ft_to_imp,ft_to_norm,test_pt)
        else: normalize_features(df_n,ft_to_imp+ft_to_norm,test_pt)
        
        features = ( [x+'_n' for x in ft_to_norm] 
                    +[x+impute*'_imp'+'_n' for x in ft_to_imp] 
                    + ft_ready )
        
        if miss_ind == 1: 
            features = features + [x + '_miss' for x in ft_to_imp]
        
        df_X1 = df_n.loc[df_n.partition!=test_pt,features]
        df_Y1 = df_n.loc[df_n.partition!=test_pt,response]
        df_X2 = df_n.loc[df_n.partition==test_pt,features]
        df_Y2 = df_n.loc[df_n.partition==test_pt,response]

        model.load_weights(wk_dir+'nn_initial_weights.h5')
        hist = model.fit(df_X1.values,df_Y1.values,epochs=e,batch_size=b_size
                  ,validation_data=(df_X2.values,df_Y2.values)
                  ,verbose=print_work)
        mse_results.append(hist.history['val_loss'][-1])
        df.loc[df.partition==test_pt,response+'_pred'] = (
                model.predict(df_X2.values,batch_size=b_size))
 
    mse_df = pd.DataFrame(mse_results,index=range(len(pts))
                ,columns=['val_mse'])
    return mse_df


def sum_results(df,val_mse_df,competitor):
    df['sq_err_'+competitor]=(df['vmax_'+competitor]-df['vmax'])**2
    df['sq_err_pred']        = (df[response+'_pred']-df[response])**2
    res = []
    for pt in range(df.partition.max()+1):
        n_obs = (df.partition == pt).sum()
        test_storms = len(df[df.partition == pt]['storm_id'].unique())
        val_mse  = val_mse_df.get_value(pt,'val_mse')
        val_mse2  = df.loc[(df.partition == pt),'sq_err_pred'].mean()
        hwrf_mse = df.loc[(df.partition == pt),'sq_err_'+competitor].mean()
        res.append((n_obs,test_storms,val_mse,val_mse2,hwrf_mse,val_mse-hwrf_mse))
    result = pd.DataFrame(res,index=val_mse_df.index,columns=[
            'n_test','n_storms_test','val_mse','val_mse_direct',competitor+'_mse','difference'])
        
    print('\nResponse '+response+'\n'+str(result))
    mean_diff = result.difference.mean()
    print('Mean difference: '+str(round(mean_diff,3))+
          ' ('+str(round(100*mean_diff/df['sq_err_'+competitor].mean(),1))+'%)') ## only valid if full dataset used
    return result
    

### EXECUTE ###
np.random.seed(rand_seed)
hf_raw = pd.read_csv(wk_dir+filename,index_col=0,parse_dates=['date']) 
hf=hf_raw[(hf_raw.vmax != -9999) & (hf_raw['vmax_'+competitor] != -9999)]
hf=hf[hf.lead_time.isin(lead_times)]
hf=kfold_partition(hf,'storm_id',10)

base_vars = list(hf.loc[1:2,'V1':'V62'])
ft_to_impute = base_vars+ ft_to_imp + (
        init_vals*list(hf.loc[1:2,'V1_t0':'V62_t0']) )
p = (1+miss_ind)*len(ft_to_impute)+len(ft_to_norm)+len(ft_ready)
nn_model = create_model(p)
nn_model.save_weights(wk_dir+'nn_initial_weights.h5')

val_mse=apply_model(hf,nn_model,ft_to_impute,ft_to_norm,ft_ready,
                    response,epochs,batch_size)
print_settings()
results=sum_results(hf,val_mse,competitor)

#hf.to_csv(path_or_buf=wk_dir+'1_model_preds.csv',index=True)
