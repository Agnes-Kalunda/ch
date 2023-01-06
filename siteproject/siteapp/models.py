from django.db import models
import plotly
import json
import os
from django.http import HttpResponse, JsonResponse
from django.shortcuts import render, redirect
import pickle
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd


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