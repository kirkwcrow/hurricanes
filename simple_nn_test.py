import numpy as np
import pandas as pd
from scipy.stats import t
from sklearn.linear_model import LinearRegression
from keras.models import Model
from keras.models import Sequential
from keras.layers import Input,Dense,Concatenate

n = 500
p = 3



#groups related features before combining
def grouped_model(feature_groups,scale): # scale determines # of interactions
    input_groups = [] # groups of features to input
    layer_1_grouped = []
    i = 0
    for g in feature_groups:
        input_groups.append(Input(len(g),))
        layer_1_grouped.append(Dense(np.ceil(scale*len(g))
                    ,kernel_initializer='normal'
                    ,activation='sigmoid') (input_groups[i]))
        i=i+1
    
    layer_2_dense = Concatenate()(layer_1_grouped)        
    layer_3 = Dense(1000,kernel_initializer='normal'
                    ,activation='sigmoid')(layer_2_dense)
    layer_4 = Dense(30, kernel_initializer='normal'
                    , activation='relu')(layer_3)
    output = Dense(1, kernel_initializer='normal',activation='linear')(layer_4)

    model = Model(input_groups,output)
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model

def build_data(groups,n_obs,p):
    # build dataframe with random values for p parameters
    df= pd.DataFrame(data=np.random.rand(n_obs,p*groups)*10
        ,index=[a for a in range(n_obs)]
        ,columns=['feature_'+str(a+1) for a in range(p)])
    df['y']   = 0
    for g in groups: 
        df['y'] = df['y'] + (df.iloc[:,g:(p+g*p)]).sum(axis=1)**(g+1) # FINISH THIS
    
#    X = df.iloc[:,0:3].values
#    Y = df['y'].values


# simple model
model = Sequential()
model.add(Dense(13, input_dim=3, kernel_initializer='normal', activation='relu'))
#model.add(Dense(100, kernel_initializer='normal', activation='sigmoid'))
model.add(Dense(1, kernel_initializer='normal',activation='linear'))

model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(X,Y,epochs=10,batch_size=15)
df['pred']=model.predict(X)
df['sq_err']=(df['pred']-df['y'])**2
#
#print(df.head())
#mse=round(df.sq_err.mean(),2)
#print('MSE: '+str(mse)+' RMSE: '+str(round(np.sqrt(mse),2)))
#print(df[['y','pred']].describe())