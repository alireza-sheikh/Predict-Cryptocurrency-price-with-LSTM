# Part_1: import necessary libraries 
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pandas_datareader as web
import datetime as dt

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras.models import Sequential

# Part_2: Getting financial data
crypto_currency = 'BTC'      # I wanna predict bitcoin price 
against_currency = 'USD'     # Getting bitcoin price against US dollar

# Setting the period time
start = dt.datetime(2016,1,1)
end = dt.datetime.now()

# Getting data from yahoo financial
data = web.DataReader(f'{crypto_currency}-{against_currency}', 'yahoo', start, end)

# Part_3: Prepare Data for testing and trainnig
scaler = MinMaxScaler(feature_range=(0, 1))                             # Scale data between zero and one
scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1,1))  # Scale 'CLose' value between -1,1

prediction_days = 60        # I wanna use past 60 days to predict next day 
future_day = 1              # The number of days in the future that we want to predict
x_train, y_train = [], []   # Create empty lists

# Fill empty lists with real data
for x in range(prediction_days, len(scaled_data)-future_day):
    x_train.append(scaled_data[x-prediction_days:x, 0])
    y_train.append(scaled_data[x+future_day, 0])

x_train, y_train = np.array(x_train), np.array(y_train)     # Converting data to numpy array 
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))      # Add another dimension

