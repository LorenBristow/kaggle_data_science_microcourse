# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 13:26:55 2019

@author: loren
"""
#NOTES#
#The model will learn and be validated based on data relating to the Melbourne housing market. 
#The model will then be applied to data relating to the Iowa housing market ('test.csv') to make predictions for competition entry.

import pandas as pd

#STEP 1 - READ IN THE TRAINING DATA - MELBOURNE DATA#

melb_data_file_path = 'input/melb_data.csv'
melb_dataframe = pd.read_csv(melb_data_file_path) 

#STEP 2 - READ IN THE TEST DATA - IOWA DATA#

iowa_data_file_path = 'input/test.csv'
iowa_dataframe = pd.read_csv(iowa_data_file_path) 

#STEP 3 - INSPECT THE DATASETS#

print('MELBOURNE data summary:\n\n', melb_dataframe.describe(),'\n\n')
print('IOWA data summary:\n\n', iowa_dataframe.describe(), '\n\n')
print('How many columns are in each dataset and what are the column names?\n')
print('There are {} columns in the MELBOURNE dataset:\n'.format(melb_dataframe.shape[1]), list(melb_dataframe),'\n\n') 
print('There are {} columns in the IOWA dataset:\n'.format(iowa_dataframe.shape[1]), list(iowa_dataframe),'\n\n')
    #Retrieving number of colums and rows in DataFrame - https://stackoverflow.com/questions/20297332/python-pandas-dataframe-retrieve-number-of-columns. 
    #dataframe.shape[0] would give number of rows. dataframe.shape with no index gives a tuple of (rows, columns). 
print('What are the datatypes by column?\n')
print('MELBOURNE column names & datatypes:\n', melb_dataframe.dtypes, '\n\n')
print('IOWA column names & datatypes:\n', iowa_dataframe.dtypes, '\n\n')
print('Which columns, if any, have missing values?\n')
print('MELBOURNE list of columns with missing values:\n', melb_dataframe.columns[melb_dataframe.isnull().any()])
print('IOWA list of columns with missing values:\n', iowa_dataframe.columns[iowa_dataframe.isnull().any()])

# dropna drops missing values (think of na as "not available")
#melbourne_data = melbourne_data.dropna(axis=0)