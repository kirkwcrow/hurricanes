import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import t

### PROGRAM PARAMETERS ###
wk_dir     = "D:\\System\\Documents\\ACADEMIC\\HF\\Data\\2018_12_21\\"

file_seq   = '1_seq_predictions.csv' # with NHC, HWRF non-null required
file_seq2  = '1_seq_predictions_more_preds.csv' # with NHC, HWRF non-null allowed


seq  = pd.read_csv(wk_dir+file_seq,index_col=0)
seq2 = pd.read_csv(wk_dir+file_seq2,index_col=0)


key_vars = ['storm_id','lead_time','date']
seq_sub = seq[key_vars+['vmax_pred_seq']].copy()
seq_sub.rename(columns={'vmax_pred_seq':'vmax_pred_seq_orig'},inplace=True)

seq2 = seq2.merge(seq_sub,on=key_vars,how='left')

seq2.to_csv(wk_dir+'seq_predictions_for_RI.csv',index=False)