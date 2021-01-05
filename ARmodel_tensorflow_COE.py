# -*- coding: utf-8 -*-
"""
Created on Fri Oct  9 14:01:01 2020

@author: pete_
"""
#Long Short Term Memory Autoregressive NN Model

#RUN ON GOOGLE COLAB FOR GPU

#This auto-regressive model is good for regression! For classification
#we instead have multiple samples of different categories (e.g. soundfiles)
# and then extract features from those (FT, SLFT, mean value etc.)

#For cross validation we can do sklearn timeseriessplit once
# the training data matrix has been made

# Univariate multi-step vector-output lstm example


#import
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import datetime as dt

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Bidirectional
from tensorflow.keras.layers import Dropout

from sklearn.model_selection import train_test_split
from sklearn.model_selection import TimeSeriesSplit

from sklearn.preprocessing import RobustScaler

from sklearn.metrics import mean_squared_error


#Import data
loc = r"C:\Users\pete_\Documents\Coding\Timeseries\COE.xls"
Excel_file = pd.ExcelFile(loc)
df = Excel_file.parse('COE data')

# correct mistakes in year
df.loc[194 , 'DATE'] = dt.datetime(2004, 2, 15, 0, 0)
df.loc[198 , 'DATE'] = dt.datetime(2004, 4, 15, 0, 0)
df.loc[202 , 'DATE'] = dt.datetime(2004, 6, 15, 0, 0)


#Automatic scaling
data = df.copy()
data = data.set_index('DATE')
scaler = RobustScaler()
target = data[['COE$']]
scaler = scaler.fit(target.to_numpy().reshape(-1,1))
target = scaler.transform(target.to_numpy().reshape(-1,1)).flatten()
target = pd.DataFrame(target,index=data.index)


#Trend can be removed by transformations,i.e. differencing the data first
#can also use other transforms e.g. log transform for exponential trends
# or combinations

# =============================================================================
# #Automatic features extraction!
# from tsfresh import extract_relevant_features
# 
# features_filtered_direct = extract_relevant_features(timeseries, y,
#                                                      column_id='id', column_sort='time')
# =============================================================================


# =============================================================================
# #Resample to check for seasonality/trends
# target.resample('Y').sum().plot()
# #Use info to create new features like target.index.dayofweek
# 
# =============================================================================

#%%

# split a univariate sequence into samples
def split_sequence(sequence, n_steps_in, n_steps_out):
    X = []
    y = []
    seq = pd.DataFrame(sequence)
    for i in range(len(seq)):
        # find the end of this pattern
        end_ix = i + n_steps_in
        out_end_ix = end_ix + n_steps_out
        # check if we are beyond the sequence
        if out_end_ix > len(seq):
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = seq.iloc[i:end_ix,:].values, seq.iloc[end_ix:out_end_ix,:].values
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)
 

# choose a number of time steps (out<test_size)
n_steps_in, n_steps_out = 12, 1
# split into samples
X, y = split_sequence(target, n_steps_in, n_steps_out)


#Add features
# Define a rolling window with Pandas, for std 
added_feat_size = 1 #values pr time point
n_std_steps = 8
if n_steps_out==1:
    data_std  = target.rolling(n_std_steps).aggregate(np.std).iloc[n_steps_in:]
else:
    data_std  = target.rolling(n_std_steps).aggregate(np.std).iloc[n_steps_in:-(n_steps_out-1)]

#Since it's only 1 value per point we add it to dim 2
#if this also was a sequence it should be added in dataframe from start
data_std = data_std.values.reshape((data_std.shape[0],added_feat_size,data_std.shape[1]))
X=np.column_stack((X,data_std))

    #This will be dim [samples, values pr point, different_features]

# reshape from [samples, timesteps] into [samples, timesteps, features]
#n_features = 1
#X = X.reshape((X.shape[0], X.shape[1], n_features))


#%%

# =============================================================================
# #Find out the best model to use
# 
# #Can also use timeseries split and plot score over time! This lets us know
# # about regions that hurt the score or find non-stationary signals
# 
# X_1 = X[:]
# y_1 = y[:]
# tscv = TimeSeriesSplit(n_splits=10)#,max_train_size=20) #Change if validation has dips over time
# for train_index, test_index in tscv.split(X_1):
#     print("TRAIN:", train_index, "TEST:", test_index)
#     X_1_train, X_1_test = X_1[train_index], X_1[test_index]
#     y_1_train, y_1_test = y_1[train_index], y_1[test_index]
#     
#     #train model
#     
#     #predict values
#     
#     #calculate score 
# =============================================================================

#%%

#Split in train,test
test_size = 10
X_train,y_train = X[:-test_size],y[:-test_size]
X_test,y_test = X[-test_size:],y[-test_size:]

#%%

# =============================================================================
# LSTM are complicated neural networks that remember "older" data
# It loads in data like RNN's in a sequential way, relevent for data 
# structured in a sequential way, ordered (time-series, language etc.)
#  LSTM is a special case of RNN
# 
# They accept an input vector x and give you an output vector y. 
# However, crucially this output vector’s contents are influenced 
# not only by the input you just fed in, but also on the entire 
# history of inputs you’ve fed in in the past. It learns to forget certain
# values later
# 
# Normal NN: take in x[0], through weights to z[0], to y[0], 
#  then x[1], through weights to z[1], to y[1]
# 
# RNN: take in x[0], through weights to z[0] and save z[0] , to y[0], 
# then x[1], through weights and add z[0] through weights to z[1] and save, to y[1]. So there is an extra set of weights between hidden layers
# going through to adjacent examples in sequence. In this way it saves
# some kind of memory of the previous example. RNN can also be stacked leading
# to even more complex structures!
# Visually, think of hidden layer nodes going back into themselves. This
# loop can be unrolled out to many connected boxes, one for each example.
# end result is that our network is structured in time so that input has to
# be ordered in time, in order to be put in to the model. 
# 
# 
# 
# This chain-like nature reveals that recurrent neural networks are 
# intimately related to sequences and lists. They’re the natural 
# architecture of neural network to use for such data!
# Just like convolutional NN (multi dim NN) are better for image data
# 
# RNN are not capable of learning long range correlations unfortunately
# LSTM is a special kind of RNN that can!
# 
# LSTMs also have this chain like structure, but the repeating module 
# has a different structure. Instead of having a single neural network 
# layer, there are four, interacting in a very special way.
# The key to LSTMs is the cell state, the horizontal line running 
# through chain. There are "gates" that forget information dependent on
# inputs and previous input, but taking this info out of the cell state.
# Example: if previous sentence has a subject and new sentence has one,
# forget previous one, use new one and remember new one.
# Complicated variants exists where gates can look at the cell state.
# 
# 
# Different if we just used linear regression/normal NN, this just uses 
# data examples as independent from each other,  i.e. we could insert data
# in a different order in principle. So it is
# not really learning the true time dynamics. Lin regression just learns
# how much in general previous values influence todays value linearly.
# NN allows for more complex non-linear functions but still generalized
# over all time. I.e.  t=3 can unfluence t=6 in some complex non-linear way
# but this will be the same for t=6 vs t=9.
# LSTM can change through time, since data is inserted sequentially!!!
# so t=3 vs t=6 can change dependent on t=2 etc. it has longer term memory
# hence the name "long short term memory" it has long term memory but, this
# gets updated once in a while, and some things will be forgotten
# =============================================================================

# Vanilla LSTM
model = Sequential()
model.add(LSTM(100, activation='relu', input_shape=(X.shape[1], X.shape[2])))
model.add(Dropout(rate=0.1)) #Dropout layer to prevent overfitting
model.add(Dense(n_steps_out))
model.compile(optimizer='adam', loss='mse')


# Other models 

# =============================================================================
# # Stacked
# model = Sequential()
# model.add(LSTM(100, activation='relu', return_sequences=True, input_shape=(X.shape[1], X.shape[2])))
# model.add(LSTM(100, activation='relu'))
# model.add(Dropout(rate=0.1)) #Dropout layer to prevent overfitting
# model.add(Dense(n_steps_out))
# model.compile(optimizer='adam', loss='mse')
# =============================================================================

# =============================================================================
# # BiDirectional
# model = Sequential()
# model.add(Bidirectional(LSTM(100, activation='relu'), input_shape=(X.shape[1], X.shape[2])))
# model.add(Dropout(rate=0.1)) #Dropout layer to prevent overfitting
# model.add(Dense(n_steps_out))
# model.compile(optimizer='adam', loss='mse')
# =============================================================================



# fit model
model.fit(X_train, y_train, epochs=300, verbose=1)

#%%
# demonstrate prediction
x_input = X_test[0:1]
y_pred_test = model.predict(x_input, verbose=0)
y_pred_train = model.predict(X_train, verbose=0)


#Converting and rescaling

#Predicted training
#Convert to dataframe
y_pred_train = pd.DataFrame(y_pred_train)
#Select out 1-day ahead predictions for the training curve
y_pred_train=y_pred_train[0]
#inverse scaling
y_pred_train = pd.Series(scaler.inverse_transform(y_pred_train.to_numpy().reshape(1,-1)).flatten())
y_pred_train.index = target.index[n_steps_in:-test_size-(n_steps_out-1)]

#Predicted test
y_pred_test = pd.Series(scaler.inverse_transform(y_pred_test.reshape(1,-1)).flatten())
y_pred_test.index = target.index[-test_size:-test_size+n_steps_out]

#True training
y_train_t= pd.Series(scaler.inverse_transform(target[:-test_size].to_numpy().reshape(-1,1)).flatten())
y_train_t.index = target[:-test_size].index

#True test
y_test_t= pd.Series(scaler.inverse_transform(target[-test_size:].to_numpy().reshape(-1,1)).flatten())
y_test_t.index = target[-test_size:].index
#%%



plt.figure(1)
plt.plot(y_train_t,label="True train",linewidth=3.0)
plt.plot(y_test_t,label="True test",linewidth=3.0)
plt.plot(y_pred_train,'o-',label="Predicted train",markersize=3)
plt.plot(y_pred_test,'o-',label="Predicted test",markersize=3)
plt.legend()

plt.show()