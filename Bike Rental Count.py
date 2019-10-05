# -*- coding: utf-8 -*-
"""
Created on Sun Aug 25 15:21:33 2019

@author: MUKHESH
"""

import matplotlib.pyplot as plt
import pandas as pd
import os
import pandas_profiling as pdpf
import webbrowser as wb
#import sklearn.ensemble.RandomForestRegressor as RF
from sklearn.model_selection import train_test_split

#Changing to working directory
os.chdir('D:/Edwisor/Bike Rental Count')
#loading the csv file data to bike_data variable
bike_data=pd.read_csv('day.csv')
#printing the profiling report
print(bike_data.shape,bike_data.columns)
print(bike_data.head(5))
#profiling_report=pdpf.ProfileReport(bike_data)
#profiling_report.to_file('output.html')
#wb.open('output.html')
#casual_bike_data=bike_data.drop(columns=['registered','count'])
#registered_bike_data=bike_data.drop(columns=['casual','count'])
#count_data=bike_data['count']
#X_train_casual=casual_bike_data.drop(columns=['casual'])
#Y_train_casual=casual_bike_data['casual']
#X_train,Y_train,X_test,Y_test=train_test_split(X_train_casual,Y_train_casual,test_size=0.2)
#model=RF()
#model.fit(X_train,Y_train)
#predictions=model.predict(X_test)
#for i in predictions:
#    print('the predicted value is {} and original value is {}'.format(predictions,Y_test))