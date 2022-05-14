import csv
import glob 
import pandas as pd
import numpy as np 
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
