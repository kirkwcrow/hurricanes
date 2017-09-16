import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


wk_dir     = "D:\\System\\Documents\\ACADEMIC\\HF\\Data\\"
filename   = 'imputation_test_data_4.xlsx'

feature_list = ['V1','V2','V3']

def impute_features(df,feature_list,test_pt):
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
    for ft in feature_list:
        df[ft+'_imp'] = tmp[ft+'_imp']
    
hf = pd.read_excel(wk_dir+filename)
impute_features(hf,feature_list,2)
print(hf)