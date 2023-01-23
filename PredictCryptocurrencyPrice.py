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

# Part_4: Create Neural Network  
model = Sequential()
# Add LSTM and Dropout and Dense layers
# The number of units is arbitary
model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=50))
model.add(Dropout(0.2))
model.add(Dense(units=1))      # We just want one output so unit is equal one

model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(x_train, y_train, epochs=25, batch_size=32)

# Part_5: Test The Model
# Setting the period time for data
test_start = dt.datetime(2022, 12, 1)
test_end = dt.datetime.now()

# Getting data from yahoo
test_data = web.DataReader(f'{crypto_currency}-{against_currency}', 'yahoo', test_start, test_end)
actual_prices = test_data['Close'].values

# Create dataset for model
total_dataset = pd.concat((data['Close'], test_data['Close']),axis =0)

# Prepare data for testing
model_inputs = total_dataset[len(total_dataset) - len(test_data) - prediction_days:].values
model_inputs = model_inputs.reshape(-1, 1)
model_inputs = scaler.fit_transform(model_inputs)

# Create empty list
x_test = []

# Fill empty list with test_data
for x in range (prediction_days, len(model_inputs)):
    x_test.append(model_inputs[x - prediction_days :x, 0])

# Convert to numpy array
x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

# The result of the test data
prediction_prices = model.predict(x_test)
prediction_prices = scaler.inverse_transform(prediction_prices)

# Part_6: visualize the resualt
plt.plot(actual_prices, color='black', label='Actual Prices')
plt.plot(prediction_prices, color='green', label='Predicted Prices')
plt.title(f'{crypto_currency} price prediction')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend(loc='upper left')
plt.show()
