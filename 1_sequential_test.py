import numpy as np
import pandas as pd
from keras.models import Model,Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers import Input,Dense,Concatenate,Dropout
from keras import regularizers

### PROGRAM PARAMETERS ###
wk_dir     = "D:\\System\\Documents\\ACADEMIC\\HF\\Data\\2018_12_21\\"
filename   = '0_clean_data.csv'
print_work = 0
rand_seed  = 466
var_to_keep = ['storm_id',
 'lead_time',
 'date',
 'vmax',
 'vmax_op',
 'vmax_nhc',
 'vmax_ivcn',
 'vmax_hwrf',
 'vmax_hwfi',
 'orig_storm_id',
 'dataset',
 'dataset_ind',
 'orig_id']


### MODEL PARAMETERS ###
ft_ready    = ['V6_x','V6_y','V8_x','V8_y','dataset_ind'] #'V6_y_miss','V8_y_miss'  #  no processing
ft_to_norm  = ['vmax_op_t0','vmax_hwfi'] # normalize only
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

def create_model_relu(p): # RE ASSIGN MODEL
    model  = Sequential()
    l_size = 65
    reg    = 0.5
    model.add(Dense(l_size, input_dim=p, kernel_initializer='normal'
                    ,kernel_regularizer=regularizers.l1(reg)
                    ,activation='relu'))
    model.add(Dense(l_size, input_dim=p, kernel_initializer='normal'
                    ,kernel_regularizer=regularizers.l1(reg)
                    ,activation='relu'))
    model.add(Dense(1, kernel_initializer='normal'
                    ,kernel_regularizer=regularizers.l1(reg)
                    ,activation='linear'))
    model.compile(loss=cost_fn, optimizer='adam')
    return model

# normalizes values based on training data
def normalizer(df,training_obs,feature_list):
    for ft in feature_list:
        train_data  = df.loc[(df[ft]!=-9999) & training_obs,ft]
        df[ft+'_n'] = ( ((df[ft]-train_data.mean())/train_data.std())*
                        (df[ft] != -9999).astype(int) ) # norm. non-missing  
    
# runs a full training iteration
def full_train(df,model,train_obs,features,response,refit = True):
    if refit: model.load_weights(wk_dir+'nn_initial_weights.h5')
    df_X1 = df.loc[train_obs,features]
    df_Y1 = df.loc[train_obs,response]
    model.fit(df_X1.values,df_Y1.values,epochs=epochs,batch_size=batch_size
              ,verbose=print_work)

# orders storms in batches for training/testing
def training_seq(df,k):
    # set up storm dataframe (storm_id, start_dt, end_dt, train_order)
    storm_df = df.groupby('storm_id').agg({'date':['min','max']})
    storm_df.columns=(['start_dt','end_dt'])
    storm_df.sort_values('start_dt',inplace=True)
    storm_df['train_order'] = np.zeros(len(storm_df),int)
    
    # get index of first 2017 storm
    first_storm_2017 = storm_df[storm_df.start_dt >= '2017-01-01'].index[0]
    first_storm_idx  = list(storm_df.index).index(first_storm_2017)
    
    # loop over storms, add in batches of size = k, or > k if storms overlap
    batch_no = 1
    batch_counter = 0
    for i in range(first_storm_idx,len(storm_df)):
        curr_start_dt = storm_df.start_dt.iat[i]
        prev_end_dt   = storm_df.end_dt.iat[i-1]
        if batch_counter >= k and curr_start_dt > prev_end_dt:
            batch_counter = 0
            batch_no = batch_no + 1
        batch_counter = batch_counter + 1
        storm_df.train_order.iat[i] = batch_no
    return storm_df
        

def sequential_test(df,model,ft_to_norm,ft_ready,response):
    normalizer(df,df.train_order==0,ft_to_norm)
    features = ft_ready + [f+'_n' for f in ft_to_norm]
    
    for i in range(df.train_order.max()):
        print('Training batch '+str(i)+'...('+str(sum(df.train_order<=i))+')')
        # train on storms so far
        full_train(df,model,df.train_order <= i,features,response)
    
        # predict next storm
        df_X1 = df.loc[df.train_order == i+1,features]
        df.loc[df.train_order == i+1,'vmax_pred_seq'] = (
                model.predict(df_X1.values,batch_size=batch_size))
    return df.loc[df.train_order > 0,'vmax_pred_seq']

### PREPARATION ###
np.random.seed(rand_seed)
hf_raw = pd.read_csv(wk_dir+filename,index_col=0,parse_dates=['date'])

hf=hf_raw[(hf_raw.vmax != -9999) 
           & (hf_raw['vmax_op_t0'] != -9999)
           & (hf_raw['vmax_hwrf'] != -9999)
           & (hf_raw['vmax_hwfi'] != -9999)
           & (hf_raw['vmax_'+competitor] != -9999)]
hf=hf[hf.lead_time.isin(lead_times)]
hf=hf.loc[hf.lead_time.isin(lead_times)]

#last_var = 'V62'
#base_vars = list(hf.loc[1:2,'V1':'V62'])

pred_subset = [1,2,4,5,7,9,14,17,19,24,32,35,37,41,48] ## SUBSET!!
base_vars = ['V'+str(v) for v in pred_subset]

ft_to_norm_all = ft_to_norm + base_vars
p = len(ft_to_norm_all)+len(ft_ready)
nn_model = create_model(p)
nn_model.save_weights(wk_dir+'nn_initial_weights.h5')

### EXECUTE ###
train_seq = training_seq(hf,k=1)
hf = hf.merge(train_seq,'left',left_on='storm_id'
              ,right_index=True)

for l in lead_times:
    print('Lead time: '+str(l))
    pred =sequential_test(hf[hf.lead_time==l].copy(),nn_model
                    ,ft_to_norm_all,ft_ready,'vmax') 
    hf.loc[(hf.lead_time==l) & (hf.train_order > 0),'vmax_pred_seq']=pred

out=hf.loc[(hf.train_order > 0),var_to_keep+['train_order','vmax_pred_seq']]

out.to_csv(path_or_buf=wk_dir+'1_seq_predictions_hwfi.csv',index=True)





