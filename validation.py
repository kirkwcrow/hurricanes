import numpy as np
import pandas as pd
import matplotlib as mpl
from keras.models import Sequential
from keras.layers import Dense

wk_dir     = "D:\\System\\Documents\\ACADEMIC\\HF validation\\Data\\"
file_1     = 'atlantic_dataset.csv'
file_2     = 'eastern_dataset.csv'

col_names  = {'Unnamed: 0':'orig_id',
              'V61':'vmax_ivcn',
              'V64':'vmax_nhc',
              'V65':'vmax_hwrf',
              'V66':'date',
              'V67':'lead_time',
              'V68':'orig_storm_id',
              'V69':'vmax'}

n_epochs = 20
b_size   = 15

#np.random.seed(1341)

# import single dataset
def import_data (filepath,dataset_id,dataset_name):
    df = pd.read_csv(filepath,dtype={'Unnamed: 0':'int'}) # read in data, first row is row number (integer)
    df['V66'] = pd.to_datetime(df.V66.astype(str),format='%Y%m%d%H') #parse date correctly
    del df['V54'] # delete column of all 0s
    del df['V63'] # may be legitimate predictor but labeling was confused, leave out to be safe
       
    df.rename(columns=col_names, inplace=True) #rename columns
    
    df = df[df.vmax != -9999] # can't use observations with missing vmax
    
    # switch IVCN, V62 column positions   
    cols=list(df)
    a,b=cols.index('vmax_ivcn'),cols.index('V62')
    cols[b],cols[a] = cols[a],cols[b]
    df=df[cols]

    # add vmax at lead_time 0 as a feature
    t0_obs = df.loc[df.lead_time == 0,['orig_storm_id','date','vmax']] # get initial vmax values
    t0_obs = t0_obs.rename(columns={'vmax':'vmax_t0'}) # rename column
    df=df.merge(t0_obs,how='left',on=['orig_storm_id','date']) # join back for each storm and date of measurement
    
    df = df[df.lead_time == 3] # keep simple: lead time 3 only for now
    
    # create unique IDs for when datasets are merged
    df['storm_id'] = df.orig_storm_id*10+dataset_id # create unique storm id
    df['dataset']  = dataset_name
    return df

# import both datasets and append them
def import_all (atl_path,pac_path):
    atl = import_data(atl_path,0,'atlantic')
    pac = import_data(pac_path,1,'pacific')
    return atl.append(pac,ignore_index=True,verify_integrity=True)

# define NN structure: 4 layers
def create_model(p):
    model = Sequential()
    model.add(Dense(1000, input_dim=p, kernel_initializer='normal',activation='sigmoid'))
    model.add(Dense(30, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, kernel_initializer='normal',activation='linear'))
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model

# fit model to training data, return fitted model
def fit_model(df_train,model,features,response):
    feature_values = df_train[features].values
    response_values = df_train[response].values
    model.fit(feature_values,response_values,epochs=n_epochs,batch_size=b_size,verbose=0)
    return model

# method to partition into test/train
def create_partition(df,by_id,test_pct):
    partition = pd.DataFrame(data=df[by_id].unique(),columns=[by_id]) # create dataframe with unique values of by_id
    rand_vector = np.random.rand(len(partition)) # create random vector of uniform[0,1], one for each by_id value
    partition['is_test'] = (rand_vector < test_pct).astype(int) # random vector determines partition
    return partition

# normalizes the training data and then applies same normalization to test data
def normalize_data(df_train,df_test,features):
    for ft in features:
        non_miss_training_data = df_train.loc[df_train[ft] != -9999,ft] # get non-missing training values of the feature
        mean = non_miss_training_data.mean() # calculate the mean of these values
        std  = non_miss_training_data.std() # calculate the st. deviation of these values
        df_train[ft+'_n'] = ((df_train[ft]-mean)/std)*(df_train[ft] != -9999) # norm. and replace with 0 if value is miss
        df_test[ft+'_n']  = ((df_test[ft] -mean)/std)*(df_test[ft]  != -9999) # norm. and replace with 0 if value is miss

# get data, partition it
hf = import_all(wk_dir+file_1,wk_dir+file_2)
partition = create_partition(hf,'storm_id',0.10) # 10% in test set to mirror 10-fold cross-validation
hf = hf.merge(partition,how='left',on='storm_id') # add column is_test to dataset

# split into train, test
hf_train = hf[hf.is_test == 0].copy()
hf_test  = hf[hf.is_test == 1].copy()

# get list of features, response
base_features = list(hf.loc[:,'V1':'V62']) # extract list of features from dataframe
all_features = base_features + ['vmax_hwrf'] #['vmax_t0'] # include initial vmax value
response = 'vmax'

# normalize
normalize_data(hf_train,hf_test,all_features)
all_features_n = [x+'_n' for x in all_features] # list of normalized features V1_n, V2_n, etc

# APPLY MODEL
n_of_features = len(all_features_n)
nn = create_model(n_of_features)
fitted_nn = fit_model(hf_train,nn,all_features_n,response)
test_predictions = fitted_nn.predict(hf_test[all_features_n].values)
hf_test['vmax_nn'] = test_predictions

## Evaluate performance
competitors = ['hwrf','ivcn','nhc']
results_list = [] # append tuples of results to this
hf_test['abs_err_nn'] = np.abs(hf_test.vmax - hf_test['vmax_nn']) 
for m in competitors:
    hf_test['abs_err_'+m] = np.abs(hf_test.vmax - hf_test['vmax_'+m]) # absolute error of difference
    non_missing_predictions = hf_test['vmax_'+m] != -9999 # shows obs where competitor made a prediction
    
    n_tot = len(hf_test)
    n_non_miss = non_missing_predictions.sum()
    n_storms = len(hf_test['storm_id'].unique())
    n_storms_non_miss = len(hf_test.loc[non_missing_predictions,'storm_id'].unique())
    competitor_mae = hf_test.loc[non_missing_predictions,'abs_err_'+m].mean() # MAE for competitor when it made prediction
    nn_mae         = hf_test.loc[non_missing_predictions,'abs_err_nn'].mean() # MAE for NN when competitor made prediction
    pct_diff       = round(((nn_mae-competitor_mae)/competitor_mae),2)        # % difference in MAE
    res = (n_tot,n_non_miss,n_storms,n_storms_non_miss,competitor_mae,nn_mae,pct_diff) # store these values
    results_list.append(res)
results_table = pd.DataFrame(data=results_list,columns=
                             ['n_tot','non_miss','n_storms','n_s_non_miss'
                              ,'comp_mae','nn_mae','pct_diff'],index=competitors)
print(results_table)

# INDIVIDUAL STORMS
hf_vis = hf_test[hf_test.vmax_ivcn != -9999].copy()
hf_vis['nn_diff']=hf_vis.abs_err_nn - hf_vis.abs_err_ivcn
storm_performance = hf_vis.groupby(['storm_id'])['nn_diff'].agg(['mean'])
#hf_vis[hf_vis.storm_id == 160].plot(x='date',y=['vmax','vmax_nn','vmax_ivcn'])