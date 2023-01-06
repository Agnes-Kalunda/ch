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