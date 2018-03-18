import numpy as np
import pandas as pd
from scipy.stats import t
from sklearn.linear_model import LinearRegression
from keras.models import Model,Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers import Input,Dense,Concatenate,Dropout
from keras import regularizers
import datetime

### PROGRAM PARAMETERS ###
wk_dir     = "D:\\System\\Documents\\ACADEMIC\\HF\\Data\\2018_03_17\\"
filename   = '0_clean_data.csv'
file_gar   = '1_garson_importance.csv'
plot_subti = 'shallow separate'
print_work = 1
rand_seed  = 466

### MODEL PARAMETERS ###
ft_ready    = ['V6_x','V6_y','V8_x','V8_y','dataset_ind'] #'V6_y_miss','V8_y_miss'  #  no processing
ft_to_norm  = ['vmax_op_t0','vmax_hwrf'] # normalize only
lead_times  = [3*(x+1) for x in range(24)]
epochs      = 40
batch_size  = 30
cost_fn     = 'mean_squared_error'
competitor  = 'nhc'
response    = 'vmax'

### FUNCTIONS ###
def create_model(p):
    model = Sequential()
    model.add(Dense(1300, input_dim=p, kernel_initializer='normal'
                    ,kernel_regularizer=regularizers.l1(0.001)
                    ,activation='sigmoid'))
    model.add(Dense(1, kernel_initializer='normal'
                    ,kernel_regularizer=regularizers.l1(0.001)
                    ,activation='linear'))
    model.compile(loss=cost_fn, optimizer='adam')
    return model

def normalizer(df,training_obs,feature_list):
    for ft in feature_list:
        train_data  = df.loc[(df[ft]!=-9999) & training_obs,ft]
        df[ft+'_n'] = ( ((df[ft]-train_data.mean())/train_data.std())*
                        (df[ft] != -9999).astype(int) ) # norm. non-missing  

def sequential_test(df,model,train_obs,ft_to_norm,ft_ready,response,e,b_size):
    normalizer(df,train_obs,ft_to_norm)
    features = ft_ready + [f+'_n' for f in ft_to_norm]
    
    # initial fit
    df_X1 = df.loc[train_obs,features]
    df_Y1 = df.loc[train_obs,response]
    model.load_weights(wk_dir+'nn_initial_weights.h5')
    model.fit(df_X1.values,df_Y1.values,epochs=e,batch_size=b_size
              ,verbose=print_work)
    
    # iterative: fit on previous storms and then predict next storm
    test_storms=df[~train_obs].groupby('storm_id').agg({'date':['min','max']})
    test_storms.columns=(['start_dt','end_dt'])
    test_storms.sort_values('start_dt',inplace=True)
    to_fit = test_storms.copy()
    for s in test_storms.index:
        print('testing... storm '+str(s))
        # fit past storms
        fit_storms=to_fit[
                to_fit.end_dt < test_storms.get_value(s,'start_dt')].index
        if len(fit_storms) > 0 :
            train_new=df.storm_id.isin(list(fit_storms))
            print('training... '+str(len(fit_storms))+ 'storms, '
                  +str(train_new.sum())+' obs')
            df_X1 = df.loc[train_new,features]
            df_Y1 = df.loc[train_new,response]
            model.fit(df_X1.values,df_Y1.values,epochs=e,batch_size=b_size
                  ,verbose=print_work)  # CHANGE EPOCHS?????????????????????????????
            to_fit.drop(list(fit_storms)) # remove storms from "to fit" once fitted
        
        # predict next storm
        test_data = df.storm_id == s
        df_X1 = df.loc[test_data,features]
        df.loc[test_data,'vmax_pred_seq'] = (
                model.predict(df_X1.values,batch_size=b_size))
        
        
### PREPARATION ###
#names  = pd.read_csv(wk_dir+'pred_names.csv',index_col=0)['Name']
np.random.seed(rand_seed)
hf_raw = pd.read_csv(wk_dir+filename,index_col=0,parse_dates=['date'])

hf=hf_raw[(hf_raw.vmax != -9999) & (hf_raw['vmax_'+competitor] != -9999) 
         &(hf_raw['vmax_op_t0'] != -9999)]
hf=hf.loc[hf.lead_time.isin(lead_times)]

last_var = 'V62'
base_vars = list(hf.loc[1:2,'V1':'V62'])
ft_to_norm_all = ft_to_norm + base_vars
p = len(ft_to_norm_all)+len(ft_ready)
nn_model = create_model(p)
nn_model.save_weights(wk_dir+'nn_initial_weights.h5')

### EXECUTE ###
sequential_test(hf,nn_model,(hf.date < '2017-01-01'),ft_to_norm_all,ft_ready,
                'vmax',epochs,batch_size) 
# to do:
# LEAD TIME RESTRICTIONS
# online training: change epoch #, need other fixes?
