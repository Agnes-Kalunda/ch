from django.db import models
import plotly as plt
import json
import os
from django.http import HttpResponse, JsonResponse
from django.shortcuts import render, redirect
import pickle
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense 
from keras.layers import Dropout


# Create your models here.


#loading the dataset
dataset_train = pd.read_csv('./CSV.csv')
dataset_train.head()

#using stock price column to train the model

training_set = dataset_train.iloc[:,1:2].values

print(training_set)
print(training_set.shape)

#normalizing the dataset
scaler = MinMaxScaler(feature_range = (0,1))
scaled_training_set = scaler.fit_transform(training_set)

scaled_training_set

#adding an x and y traains data structures

x_train =[]
y_train = []

for i in range(60, 1200):
    x_train.append(scaled_training_set[i-60:i, 0])
    y_train.append(scaled_training_set[i, 0])

x_train = np.array(x_train)
y_train =np.array(y_train)

print(x_train.shape)
print(y_train)

#reshaping the data
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
x_train.shape

regressor = Sequential()

regressor.add(LSTM(units = 50, return_sequences= True, input_shape = (x_train.shape[1], 1)))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units =50, return_sequences =True))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units =50, return_sequences= True))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units= 50))
regressor.add(Dropout(0.2))

regressor.add(Dense(units=1))

#fitting the model

regressor.compile(optimizer = 'Adam', loss = 'mean_squared_error')
regressor.fit(x_train, y_train, epochs=100, batch_size=32)


#preparing the model input
dataset_total = pd.concat((dataset_train['Open'], dataset_test['Open']), axis =0)
inputs = dataset_total[len(dataset_total)- len(dataset_test)-60:].values

inputs = inputs.reshape(-1,1)
inputs = scaler.transform(inputs)

x_test = []
for i in range(60,80):
    x_test.append(inputs[i-60:i,0])
x_test = np.array(x_test)
x_test= np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

#predicting the january 2023 stock prices

predicted_stock_price = regressor.predict(x_test)
predicted_stock_price= scaler.inverse_transform(predicted_stock_price)

#ploting the data

plt.plot(actual_stock_price, color='red', label = 'Actual Google Stock Prices')
plt.plot(predicted_stock_price, color ='blue', label = 'Predicted_Stock_Prices')
plt.title('Google Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Stock prices')
plt.legend()