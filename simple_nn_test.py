import numpy as np
import pandas as pd

#import keras
#import sklearn
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

n = 500
p = 3

# build dataframe with random values for p parameters
df= pd.DataFrame(data=np.random.rand(n,p)*10
    ,index=[a for a in range(n)]
    ,columns=['feature_'+str(a+1) for a in range(p)])
df['y']   = (df.iloc[:,0:p]).sum(axis=1)

X = df.iloc[:,0:3].values
Y = df['y'].values


# simple model
model = Sequential()
model.add(Dense(13, input_dim=3, kernel_initializer='normal', activation='relu'))
#model.add(Dense(100, kernel_initializer='normal', activation='sigmoid'))
model.add(Dense(1, kernel_initializer='normal',activation='linear'))

model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(X,Y,epochs=10,batch_size=15)
df['pred']=model.predict(X)
df['sq_err']=(df['pred']-df['y'])**2

print(df.head())
mse=round(df.sq_err.mean(),2)
print('MSE: '+str(mse)+' RMSE: '+str(round(np.sqrt(mse),2)))
print(df[['y','pred']].describe())