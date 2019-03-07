# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 13:26:55 2019

@author: loren
"""

import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error

#STEP 1 - READ IN THE TRAINING DATA - MELBOURNE DATA#

melb_data_file_path = 'input/melb_data.csv'
melb_dataframe = pd.read_csv(melb_data_file_path) 

#STEP 2 - READ IN THE TEST DATA - IOWA DATA#

iowa_data_file_path = 'input/train.csv'
iowa_dataframe = pd.read_csv(iowa_data_file_path) 

#STEP 3 - INSPECT THE DATASETS#

pd.set_option('display.max_columns', None) #setting to allow all columns to be seen in console for inspection. 
print('MELBOURNE data summary:\n\n', melb_dataframe.describe(),'\n\n')
print('IOWA data summary:\n\n', iowa_dataframe.describe(), '\n\n')

#the below block is not necessary as the variable explorer shows column count and names - this was just for practice.
    #print('How many columns are in each dataset and what are the column names?\n')
    #print('There are {} columns in the MELBOURNE dataset:\n'.format(melb_dataframe.shape[1]), list(melb_dataframe),'\n\n') 
    #print('There are {} columns in the IOWA dataset:\n'.format(iowa_dataframe.shape[1]), list(iowa_dataframe),'\n\n')
    #Retrieving number of colums and rows in DataFrame - https://stackoverflow.com/questions/20297332/python-pandas-dataframe-retrieve-number-of-columns. 
    #dataframe.shape[0] would give number of rows. dataframe.shape with no index gives a tuple of (rows, columns). 

#datatypes
print('What are the datatypes by column?\n')
print('MELBOURNE column names & datatypes:\n', melb_dataframe.dtypes, '\n\n')
print('IOWA column names & datatypes:\n', iowa_dataframe.dtypes, '\n\n')

#missing values
print('Which columns, if any, have missing values?\n')
print('MELBOURNE: list of columns with missing values:\n', melb_dataframe.columns[melb_dataframe.isnull().any()], '\n\n')
print('IOWA list of columns with missing values:\n', iowa_dataframe.columns[iowa_dataframe.isnull().any()], '\n\n')

#STEP 4 - INITIAL DATA CLEANING/PREPARATION#

#For the first version of this beginner ML model columns with missing values are simply dropped.
melb_dataframe_missing_values_dropped = melb_dataframe.dropna(axis=0) #(axis=0) is indicating that columns should be dropped rather than rows (axis=1)

#STEP 5 - SELECT PREDICTION TARGET (y) AND FEATURES (X)#

#Define the prediction target 
#Inspect columns to identify the name of the price column in the dataset.
print(iowa_dataframe.columns,'\n') #.columns presents the columns headers as an index. 
#The price column in this dataset is called 'SalePrice'.
y = iowa_dataframe.SalePrice #defines the target.
print('Prices of first 5 houses in Iowa dataset:\n', y.head(), '\n') #to see the first few rows of 'y' for inspection.
    
#List the features by column name and then assign the features data to 'X'. Features are the inputs to the model. 
iowa_features = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']
X = iowa_dataframe[iowa_features]   
print('Features of first 5 houses in the Iowa dataset:\n', X.head(), '\n') 

#STEP 6 - DEFINE/SPECIFY & FIT THE MODEL#

#Using the scikit-learn library.
#Specify the model - The first version of the model is a decision tree model. 
iowa_house_price_model = DecisionTreeRegressor(random_state=1) #random state is specified to ensure the same results with each run ie. randomness is taken out. The number chose doesn't materially impact quality.
#Fit the model
iowa_house_price_model.fit(X, y)
print('Model type and specifications:\n', iowa_house_price_model,'\n\n')

#STEP 7 - MAKE PREDICTIONS#

predicted_house_prices = iowa_house_price_model.predict(X)
print('Predictions for the first 5 houses in the IOWA dataset are:\nFeatures of the houses:\n',X.head(), '\n',\
      'Predicted prices of the houses:\n', iowa_house_price_model.predict(X.head()),'\n\n') #need to understand why I can't use .head with the variable 'predicted_house_prices'.

#STEP 7 - MODEL VALIDATION#

#'Mean absolute error' (MAE) is used as a single metric by which to assess predictove accuracy ie. model quality. 
#The prediction error per house: error = actual price - predicted price. Take the mean of these as absolute values to get MAE. Smaller MAE is better MAE!
print('The mean absolute error (MAE) of the model is:\n', mean_absolute_error(y,predicted_house_prices))





