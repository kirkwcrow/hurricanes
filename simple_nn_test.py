import numpy as np
import pandas as pd
from scipy.stats import t
from sklearn.linear_model import LinearRegression
from keras.models import Model
from keras.models import Sequential
from keras.layers import Input,Dense,Concatenate

# PARAMETERS
n = 500
p = 4
g = 3

n_epochs = 100
verb = 0

# FUNCTIONS
def build_data(n,p,groups,test): # test dataset
    i = 1 # important parameters
    df= pd.DataFrame(data=np.random.rand(n,p*groups+i)
        ,index=[a for a in range(n)])
    df['y'] = 0
    for g in range(groups):
        sum_group = (df.iloc[:,g*p+1:g*p+5]).product(axis=1)
        apply_important_term = sum_group #* df.iloc[:,0] * 10
        df['group_'+str(g)] = apply_important_term.apply(lambda x: np.log(x))
        df['y'] = df['y'] + df['group_'+str(g)]
    df['test'] = (np.random.rand(n) < test)
    return df

def simple_model(p):
    model = Sequential()
    model.add(Dense(100, input_dim=p, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, kernel_initializer='normal',activation='linear'))
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model

#groups related features before combining
def grouped_model(feature_groups,scale): # scale determines # of interactions
    input_groups = [] # groups of features to inp
    layer_1_grouped = []
    i = 0
    for g in feature_groups:
        gsize = int(np.ceil(scale*len(g)))
        input_groups.append(Input((len(g),))) #inputs 
        layer_1_grouped.append(Dense(gsize
                    ,kernel_initializer='normal'
                    ,activation='sigmoid'
                    ,name='feature_group_'+str(i)) (input_groups[i]))
        i=i+1
    
    layer_2_dense = Concatenate()(layer_1_grouped)        
#    layer_3 = Dense(300,kernel_initializer='normal'
#                    ,activation='sigmoid')(layer_2_dense)
#    layer_4 = Dense(30, kernel_initializer='normal'
#                    , activation='relu')(layer_3)
    output = Dense(1, kernel_initializer='normal'
                   ,activation='linear'
                   ,name='final_output')(layer_2_dense) #4

    model = Model(input_groups,output)
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model #model.summary() to see


# EXECUTE
np.random.seed(47)
df = build_data(100,p,g,0.2)
df.to_clipboard()
features = list(df)[0:p*g+1]
X = df.loc[df.test == 0,features].values
Y = df.loc[df.test == 0]['y'].values
X_2 = df.loc[df.test,features].values
Y_2 = df.loc[df.test]['y'].values

# GROUPED NEURAL NET

ft_groups = [[0],[1,2,3,4],[5,6,7,8],[9,10,11,12]]
inputs_train = []
inputs_all   = []
for gp in ft_groups:
    inputs_train.append(df.loc[df.test==0][gp].values)
    inputs_all.append(df[gp].values)
    
nn = grouped_model(ft_groups,5)
nn.fit(inputs_train,Y,epochs=n_epochs,batch_size=15,verbose=verb) #,validation_data=(inputs_all,Y_2))
df['pred'] =nn.predict_on_batch(inputs_all)
df['abs_err']=np.abs(df['pred']-df['y'])

print(df.loc[df.test,'abs_err'].mean())


nn_simple = simple_model(len(X[0]))
nn_simple.fit(X,Y,epochs=n_epochs,batch_size=15,validation_data=(X_2,Y_2),verbose=verb)

df['pred']=nn_simple.predict(df[features].values)
df['abs_err']=np.abs(df['pred']-df['y'])

print(df.loc[df.test,'abs_err'].mean())
