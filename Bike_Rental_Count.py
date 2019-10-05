# -*- coding: utf-8 -*-
"""
Created on Thu Aug 29 10:59:38 2019

@author: MUKHESH
"""
#loading required libraries
import pandas as pd
import numpy as np
from  sklearn.tree import DecisionTreeRegressor as DTR,export_graphviz
from sklearn.ensemble import RandomForestRegressor as RFT
from sklearn.externals.six import StringIO
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import statsmodels.api as sm
import os
import datetime
import pydotplus
from IPython.display import Image
from subprocess import call
import seaborn as sn

#setting working directory
os.chdir('D:/Edwisor/Bike Rental Count')

#reading data from csv files
bike_data=pd.read_csv('day.csv',na_values=['NA', ' '])

#printing the sample data
print(bike_data.head(5))

#knowing the structure of bike_data
print(bike_data.info())

#Expolartory analysis

#dropping variables is instant because which does'nt add any information
bike_data.drop(columns=['instant'],inplace=True)

#Univariate Analysis
bike_data.hist(grid=False)

#As exploration we can find there are three dependent variable casual,registered,count
#where by summing up casual and registered gives count
#As expected, mostly working days and variable holiday and weekday is also showing a similar inference. You can use the code above to look at the distribution in detail.
#Variables temp, atemp, humidity and windspeed  looks naturally distributed. which are normalised

#converting variable into proper data type
bike_data['season']=bike_data['season'].astype('category')
bike_data['yr']=bike_data['yr'].astype('category')
bike_data['mnth']=bike_data['mnth'].astype('category')
bike_data['holiday']=bike_data['holiday'].astype('category')
bike_data['weekday']=bike_data['weekday'].astype('category')
bike_data['workingday']=bike_data['workingday'].astype('category')
bike_data['weathersit']=bike_data['weathersit'].astype('category')
#extracting the  day from dteday column
bike_data['dteday']=pd.to_datetime(bike_data['dteday'])
bike_data['dteday']=bike_data['dteday'].dt.day
#converting extracted dteday into category variable
bike_data['dteday']=bike_data['dteday'].astype('category')
#Missing values Analysis
Missing_values=pd.DataFrame(bike_data.isnull().sum())
#setting Missing value count in column
Missing_values.columns=['Missing value count']
#where we did'nt find any missing values

#here we can make some hypothesis testing from data
#where casual users increases when weekend

plt.bar(bike_data['weathersit'],bike_data['casual'])
plt.xlabel('weather')
plt.ylabel('count of casual users')
plt.title('casual user analysis on weather')
plt.show()

plt.bar(bike_data['weathersit'],bike_data['registered'])
plt.xlabel('weather')
plt.ylabel('count of registered users')
plt.title('registered user analysis on weather')
plt.show()

#at seeing bike users incresing over weekends
plt.bar(bike_data['weekday'],bike_data['casual'])
plt.xlabel('weekday')
plt.ylabel('count of casual users')
plt.title('casual user analysis on weekday')
plt.show()

plt.bar(bike_data['weekday'],bike_data['registered'])
plt.xlabel('weekday')
plt.ylabel('count of registered users')
plt.title('registered user analysis on weather')
plt.show()

#outlier Analysis
#saving numeric values#
cnames=["temp","atemp","hum","windspeed"]
#ploting boxplotto visualize outliers#
plt.boxplot(bike_data['temp'])
plt.boxplot(bike_data['atemp'])
plt.boxplot(bike_data['hum'])
plt.boxplot(bike_data['windspeed'])
#where we found outliers in hum and windspeed but we neglecting those because of those natural outiers.

#-------------------------------#feature engineering----------------------------------------
#we can create some feature like temp_regr,temp_casual
#we are using decision tree for best split for creating categories for both casual and registered variables
DT_tree_casual=DTR(min_samples_split=2,max_depth=2).fit(bike_data.iloc[:,8:9],bike_data['registered'])
#for visualizing the tree we exporting into dot_data variable where we given outfile=None so it will export as string in dot_data
dot_data=export_graphviz(DT_tree_casual, out_file=None,feature_names=['temp'] , filled=True, rounded=True,special_characters=True)
#creating pydotplus object for graphical representation
graph=pydotplus.graph_from_dot_data(dot_data)
#for visualizing the decision tree
Image(graph.create_png())

#by analyzing the decision tree we are creating variable temp_regr
bike_data['temp_regr']=0
bike_data['temp_regr'][bike_data.temp<0.27]=1
bike_data['temp_regr'][bike_data.temp>=0.51]=2
bike_data['temp_regr'][(bike_data.temp>=0.27) & (bike_data.temp<0.43)]=3
bike_data['temp_regr'][(bike_data.temp>=0.43) & (bike_data.temp<0.51)]=4
bike_data['temp_regr']=bike_data['temp_regr'].astype('category')

#we are using decision tree for best split for creating categories for both casual and registered variables
DT_tree_registered=DTR(min_samples_split=2,max_depth=2).fit(bike_data.iloc[:,8:9],bike_data['casual'])
#for visualizing the tree we exporting into dot_data variable where we given outfile=None so it will export as string in dot_data
dot_data=export_graphviz(DT_tree_registered, out_file=dot_data,  
                filled=True, rounded=True,
                special_characters=True)
#creating pydotplus object for graphical representation
graph=pydotplus.graph_from_dot_data(dot_data.getvalue())
#for visualizing the decision tree
Image(graph.create_png())
#by analyzing the decision tree we are creating variable temp_casual
bike_data['temp_casual']=0
bike_data['temp_casual'][bike_data.temp<0.32]=1
bike_data['temp_casual'][bike_data.temp>=0.51]=2
bike_data['temp_casual'][(bike_data.temp>=0.32) & (bike_data.temp<0.42)]=3
bike_data['temp_casual'][(bike_data.temp>=0.42) & (bike_data.temp<0.51)]=4
bike_data['temp_casual']=bike_data['temp_casual'].astype('category')


#--------------------------------#Feature Selection----------------------------------------
#taking corretion between numerical variables
cov=bike_data.corr()
#printing the correlation array
print(cov)
#where visualizing the correlation plot
f,ax=plt.subplots(figsize=(12,8))
sn.heatmap(cov,mask=np.zeros_like(cov,dtype=np.bool),cmap=sn.diverging_palette(220,10,as_cmap=True),square=True,ax=ax)


#removing temp variable from bike_data because atemp and temp has highly correlated
#where priting the correlation between the variables
#as you find there is high correlation between the temp and atemp varibales
#where after creating varaible we are deleting the temp variable
bike_data.drop(columns=['temp'],inplace=True)


#sampling where we spliting the data into train and test split in 80:20 ratio
train_data,test_data=train_test_split(bike_data,test_size=0.2)
train_data['casual']=np.log(train_data['casual'])
train_data['registered']=np.log(train_data['registered'])
#Modelling
#where here we have three  dependent variables
#we have to apply model two times to get the two dependent variables like casual and registered and total gives you the cnt
#where casual and registered is little skewed so we apply log transformation on both variables

#--------------------------------------#Decision Tree--------------------------------------------
#model to predict casual variable

casual_DT=DTR().fit(train_data.iloc[:,[0,1,2,3,4,5,6,7,8,9,10,15]],train_data.iloc[:,11])
predictions_casual_DT=casual_DT.predict(test_data.iloc[:,[0,1,2,3,4,5,6,7,8,9,10,15]])
predictions_casual_DT=np.exp(predictions_casual_DT)

#model to predict registered variable
registerd_DT=DTR().fit(train_data.iloc[:,[0,1,2,3,4,5,6,7,8,9,10,14]],train_data.iloc[:,12])
predictions_registered_DT=registerd_DT.predict(test_data.iloc[:,[0,1,2,3,4,5,6,7,8,9,10,14]])
predictions_registered_DT=np.exp(predictions_registered_DT)

#sum of predictions of casual variable  and registered will be give predictions of total count
predictions_DT=predictions_casual_DT+predictions_registered_DT



#-------------------------------#RandomForrest#--------------------------------------------------
#model to predict casual variable
casual_RF=RFT().fit(train_data.iloc[:,[0,1,2,3,4,5,6,7,8,9,10,15]],train_data.iloc[:,11])
predictions_casual_RF=casual_RF.predict(test_data.iloc[:,[0,1,2,3,4,5,6,7,8,9,10,15]])
predictions_casual_RF=np.exp(predictions_casual_RF)

#model to predict registered variable
registerd_RF=RFT().fit(train_data.iloc[:,[0,1,2,3,4,5,6,7,8,9,10,14]],train_data.iloc[:,12])
predictions_registerd_RF=registerd_RF.predict(test_data.iloc[:,[0,1,2,3,4,5,6,7,8,9,10,14]])
predictions_registerd_RF=np.exp(predictions_registerd_RF)

#sum of predictions of casual variable  and registered will be give predictions of total count
predictions_RF=predictions_casual_RF+predictions_registerd_RF



#------------------------------#linear regression-------------------------------------------------
#where for categorical variable dummy variables are created
#where creating dummy variables for categorical variables to be used in linear regression
#taking category variable in variable
catnames=['dteday','season','yr','mnth','holiday','weekday','workingday','weathersit','temp_casual','temp_regr']

cnames.remove('temp')
cnames=['casual','registered','cnt']+cnames
bike_linear_regression=bike_data[cnames]
#creating dummy variables
for i in catnames:
    temp=pd.get_dummies(bike_data[i],prefix=i)
    bike_linear_regression=bike_linear_regression.join(temp)
    
#sampling the data into train and test data
train_lr_data,test_lr_data=train_test_split(bike_linear_regression,test_size=0.2)
#where the casual and registered variables are skewed so i did log transformation
train_lr_data['casual']=np.log(train_lr_data['casual'])
train_lr_data['registered']=np.log(train_lr_data['registered'])
#model to predict casual variable
casual_lg=sm.OLS(train_lr_data.iloc[:,0],train_lr_data.iloc[:,3:73]).fit()
predictions_casual_lg=casual_lg.predict(test_lr_data.iloc[:,3:73])
predictions_casual_lg=np.exp(predictions_casual_lg)
print(casual_lg.summary())


#model to predict registered variable
registerd_lg=sm.OLS(train_lr_data.iloc[:,1],train_lr_data.iloc[:,list(range(3,69))+[73,74,75,76]]).fit()
predictions_registerd_lg=registerd_lg.predict(test_lr_data.iloc[:,list(range(3,69))+[73,74,75,76]])
predictions_registerd_lg=np.exp(predictions_registerd_lg)

predictions_lg=predictions_casual_lg+predictions_registerd_lg

#function for evaluation
def mape(y,y_pred):
    mape=np.mean(np.abs((y-y_pred)/y))*100
    return mape


#checking mean absolute persentage error 
#for random forrest
print(mape(test_data.iloc[:,13],predictions_RF))#-->mape=13.39
#for linear regression
print(mape(test_data.iloc[:,13],predictions_lg))#--->mape=17.29
#for decision tree
print(mape(test_data.iloc[:,13],predictions_DT))#--->mape=18.28044

#-------------Hyperparameter Tuning------------------
#creating variable for options for hyperparameter tuning
parameters={'max_features':[1,2,3,4,5,6,7,8,9,10],'n_estimators':[50,100,150,200,250,300,350]}
#creating object for getting the best parameters
grid=GridSearchCV(RFT(),param_grid=parameters,cv=10)
#training for getting best parameters for casual variable
grid.fit(train_data.iloc[:,[0,1,2,3,4,5,6,7,8,9,10,15]],train_data.iloc[:,11])

print(grid.best_score_)
#printing the best parameters
print(grid.best_params_)

#creating object for getting the best parameters
grid=GridSearchCV(RFT(),param_grid=parameters,cv=10)
#training for getting best parameters for registered variable
grid.fit(train_data.iloc[:,[0,1,2,3,4,5,6,7,8,9,10,14]],train_data.iloc[:,12])

print(grid.best_score_)
#printing the best parameters
print(grid.best_params_)

#train again and check if accuarcy is increase or not
#model to predict casual variable
casual_RF=RFT(max_features=6, n_estimators= 250).fit(train_data.iloc[:,[0,1,2,3,4,5,6,7,8,9,10,15]],train_data.iloc[:,11])
predictions_casual_RF=casual_RF.predict(test_data.iloc[:,[0,1,2,3,4,5,6,7,8,9,10,15]])
predictions_casual_RF=np.exp(predictions_casual_RF)

#model to predict registered variable
registerd_RF=RFT(max_features=4, n_estimators= 100).fit(train_data.iloc[:,[0,1,2,3,4,5,6,7,8,9,10,14]],train_data.iloc[:,12])
predictions_registerd_RF=registerd_RF.predict(test_data.iloc[:,[0,1,2,3,4,5,6,7,8,9,10,14]])
predictions_registerd_RF=np.exp(predictions_registerd_RF)

#sum of predictions of casual variable  and registered will be give predictions of total count
predictions_RF=predictions_casual_RF+predictions_registerd_RF

#printing the mape value 
print(mape(test_data.iloc[:,13],predictions_RF))#-->12.51
#where we can see there is increase in accuracy

#creating new variable in test_data for casual prediction
test_data['casual_prediction_cnt']=predictions_casual_RF
#creating new variable in test_data for casual prediction
test_data['registered_prediction_cnt']=predictions_registerd_RF
#creating new variable in test_data for prediction
test_data['prediction_cnt']=predictions_RF
#writing the data into new csv file
test_data.to_csv('bike_rental_prediction_python.csv',index=False)


















