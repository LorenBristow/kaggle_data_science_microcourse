# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 13:26:55 2019

@author: loren
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error

#STEP 1 - READ IN DATA#

iowa_data_file_path = 'input/train.csv'
iowa_dataframe = pd.read_csv(iowa_data_file_path) 

#STEP 2 - INSPECT THE DATA#

pd.set_option('display.max_columns', None) #setting to allow all columns to be seen in console for inspection. 
print('IOWA data summary:\n\n', iowa_dataframe.describe(), '\n\n')

#datatypes
print('What are the datatypes by column?\n')
print('IOWA column names & datatypes:\n', iowa_dataframe.dtypes, '\n\n')

#missing values
print('Which columns, if any, have missing values?\n')
print('IOWA list of columns with missing values:\n', iowa_dataframe.columns[iowa_dataframe.isnull().any()], '\n\n')

#STEP 3 - DATA CLEANING & PREPARATION#

#For the first version of this beginner ML model columns with missing values are simply dropped.
iowa_dataframe_missing_values_dropped = iowa_dataframe.dropna(axis=1) #(axis=1) is indicating that columns should be dropped rather than rows (axis=0)

#STEP 4 - SELECT PREDICTION TARGET (y) AND FEATURES (X), AND SPLIT DATA INTO TRAIN AND VALIDATION PORTIONS#

#Define the prediction target 
#Inspect columns to identify the name of the price column in the dataset.
print('List of columns after missing values have been removed:\n', iowa_dataframe_missing_values_dropped.columns,'\n') #.columns presents the columns headers as an index. 
#The price column in this dataset is called 'SalePrice'.
y = iowa_dataframe.SalePrice #defines the target.
print('Prices of first 5 houses in Iowa dataset:\n', y.head(), '\n') #to see the first few rows of 'y' for inspection.
    
#List the features by column name and then assign the features data to 'X'. Features are the inputs to the model. 
iowa_features = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']
X = iowa_dataframe_missing_values_dropped[iowa_features]   
print('Features of first 5 houses in the Iowa dataset:\n', X.head(), '\n') 

#Split data into separate training and validation datasets for features and for prediction target. 
train_X, validation_X, train_y, validation_y = train_test_split(X, y, random_state = 0) #the split is random so setting random state ensure the same split for every run. 

#STEP 5 - DEFINE/SPECIFY & FIT THE MODEL#

#Using the scikit-learn library.
#Specify the model - The first version of the model is a decision tree model. 
iowa_house_price_model = DecisionTreeRegressor(random_state=1) #random state is specified to ensure the same results with each run ie. randomness is taken out. The number chose doesn't materially impact quality.
#Fit the model
iowa_house_price_model.fit(train_X, train_y)
print('Model type and specifications:\n', iowa_house_price_model,'\n\n')

#STEP 6 - MAKE PREDICTIONS#

predicted_house_prices = iowa_house_price_model.predict(validation_X)
print('Predictions for the first 5 houses in the IOWA dataset are:\nFeatures of the houses:\n',validation_X.head(), '\n',\
      'Predicted prices of the houses:\n', iowa_house_price_model.predict(validation_X.head()),'\n\n') #need to understand why I can't use .head with the variable 'predicted_house_prices'.

#STEP 7 - MODEL VALIDATION#

#'Mean absolute error' (MAE) is used as a single metric by which to assess predictove accuracy ie. model quality. 
#The prediction error per house: error = actual price - predicted price. Take the mean of these as absolute values to get MAE. Smaller MAE is better MAE!
print('The mean absolute error (MAE) of the model is:\n', round(mean_absolute_error(validation_y,predicted_house_prices),3))





