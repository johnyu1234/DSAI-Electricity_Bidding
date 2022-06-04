MARKET_PRICE = 2.5256
import csv
import pandas as pd
import numpy as np
from dateutil import parser
from datetime import timedelta # to add time into current
from tensorflow.keras.models import load_model
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import LSTM
from keras.layers import Dropout

# You should not modify this part.
def config():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--consumption", default="./sample_data/consumption.csv", help="input the consumption data path")
    parser.add_argument("--generation", default="./sample_data/generation.csv", help="input the generation data path")
    parser.add_argument("--bidresult", default="./sample_data/bidresult.csv", help="input the bids result path")
    parser.add_argument("--output", default="output.csv", help="output the bids path")

    return parser.parse_args()

def create_date(starting_date):
    date = list()
    for i in range(24):
        starting_date = starting_date + timedelta(hours=1)
        date.append(starting_date)

    # for i in date:
        # print(i)
    return date

def prediction(data_csv,model):
    test_data = data_csv.iloc[:,1]
    np_test = np.array(test_data)
    # print(np_test.shape)
    np_test = np.reshape(np_test,(1,np_test.shape[0],1))
    # print(np_test.shape)
    ans = model.predict(np_test)
    ans = ans.reshape(-1) 
    return ans

def input_data(filename,data_type):
    # reading input.csv and converting into input for model prediction
    df = pd.read_csv(filename)
    X = []
    X.append(df.loc[:,data_type].tolist())
    X = np.array(X)
    X = np.reshape(X, (X.shape[0],X.shape[1],1))
    return X

def output(path, data):
    import pandas as pd

    df = pd.DataFrame(data, columns=["time", "action", "target_price", "target_volume"])
    df.to_csv(path, index=False)

    return

def action(buy, sell, gen, use, trade_price):
    a = (buy-sell) + gen
    b = a - use
    if a>=0:
        A = (buy-sell) * trade_price
    else:
        A = (-1) * gen * trade_price
    if b>=0:
        B = 0
    else:
        B = -1 * b * MARKET_PRICE
    return (A+B)

if __name__ == "__main__":
    args = config()
    data = []
    hour = []
    gen = []
    con = []
    consumption = pd.read_csv(args.consumption)
    generation = pd.read_csv(args.generation)
    # create date
    generate_date = parser.parse(consumption.iloc[-1,0]) # last hour
    hour = create_date(generate_date)
    # for i in hour:
        # print(i)
    model_con = load_model('consumption.h5', compile = False)
    model_gen = load_model('generation.h5',compile = False)
    con = prediction(consumption,model_con)
    gen = prediction(generation,model_gen)
    gen[gen<0] = 0
    con[con<0] = 0
    # print(gen)
    con = con.tolist() 
    gen = gen.tolist()
    # Decide to buy or sell
    for i in range(len(hour)):
        val = gen[i] - con[i]
        if(val > 0):
            # print(gen[i]) 
            # sell_unit = round((0.9*gen[i]), 2)
            # sell_price = action(0, sell_unit, gen[i], con[i], sell_unit)
            for a in range(int(val)):
                data.append([hour[i], 'sell', 2.2, 1])
        else:
            for a in range(abs(int(val))):            
                data.append([hour[i], 'buy', 2.5, 1])

    output(args.output, data)