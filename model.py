import csv
import glob 
import pandas as pd
import numpy as np 

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import LSTM

path = '/Users/johns/Desktop/Spring 2022/DSAI-Electricity_Bidding/training_data'

csv_files = glob.glob(path+"/*.csv")

train_X_con = []
train_Y_con = []

train_X_gen = []
train_Y_gen = []

for fname in csv_files:
    df = pd.read_csv(fname)
    for i in range(int(df.shape[0] / 24 - 7)):
        train_X_con.append(df.loc[i * 24 : (i + 7) * 24 - 1, 'consumption'].tolist())
        train_Y_con.append(df.loc[(i + 7) * 24 : (i + 8) * 24 - 1, 'consumption'].tolist())
        
        train_X_gen.append(df.loc[i * 24 : (i + 7) * 24 - 1, 'generation'].tolist())
        train_Y_gen.append(df.loc[(i + 7) * 24 : (i + 8) * 24 - 1, 'generation'].tolist())

train_X_con = np.array(train_X_con)
train_Y_con = np.array(train_Y_con)
train_X_gen = np.array(train_X_gen)
train_Y_gen = np.array(train_Y_gen)


print(train_X_con.shape)
print(train_Y_con.shape)


n_timesteps, n_features, n_outputs = train_X_con.shape[1], train_X_con.shape[2], train_Y_con.shape[1]
# define model
model = Sequential()
model.add(LSTM(200, activation='relu', input_shape=(n_timesteps, n_features)))
model.add(Dense(100, activation='relu'))
model.add(Dense(n_outputs))
model.compile(loss='mse', optimizer='adam')
# fit network
final = model.fit(train_X_con, train_Y_con, epochs=epochs, batch_size=batch_size, verbose=verbose)
# model referencing 
# https://machinelearningmastery.com/how-to-develop-lstm-models-for-multi-step-time-series-forecasting-of-household-power-consumption/


