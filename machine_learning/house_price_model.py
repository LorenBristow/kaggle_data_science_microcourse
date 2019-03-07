# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 13:26:55 2019

@author: loren
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor

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
train_X, validation_X, train_y, validation_y = train_test_split(X, y, random_state = 1) #the split is random so setting random state ensure the same split for every run. 

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

#STEP 7 - MODEL VALIDATION (& IMPROVEMENTS & VALIDATION &...)#

#'Mean absolute error' (MAE) is used as a single metric by which to assess predictove accuracy ie. model quality. 
#The prediction error per house: error = actual price - predicted price. Take the mean of these as absolute values to get MAE. Smaller MAE is better MAE!
mae_decision_tree_default = mean_absolute_error(validation_y,predicted_house_prices)
print('The mean absolute error (MAE) of predictions by this model\n(with default decision tree specifications and run on validation data only):\n', round(mae_decision_tree_default), '\n')

#To improve prediction accuracy, the depth of the decision tree can be adjusted using the 'max_leaf_nodes argument'. 
#More leaves tends towards overfitting, fewer leaves towards underfitting.
#A function to specify, fit and rerun the model and output MAE given different tree depths:
def specify_fit_predict_validate(max_leaf_nodes, train_X, validation_X, train_y, validation_y):
        iowa_house_price_model_adjusting_max_leaves = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=1)
        iowa_house_price_model_adjusting_max_leaves.fit(train_X, train_y)
        predicted_house_prices_adjusting_max_leaves = iowa_house_price_model_adjusting_max_leaves.predict(validation_X)
        mae = mean_absolute_error(validation_y, predicted_house_prices_adjusting_max_leaves)
        return(mae)

#A for loop to test various decision tree depths(max_leaf_nodes) and return optimal number of nodes:
max_leaves_for_consideration = [5, 25, 50, 100, 250, 500]
mae_for_comparison_list = []
print('Initial testing of different numbers of leaf nodes results in the following:\n')
for max_leaf_nodes in max_leaves_for_consideration:
    mae_for_comparison = specify_fit_predict_validate(max_leaf_nodes, train_X, validation_X, train_y, validation_y)
    print('leaf nodes: {},    mae: {}'.format(max_leaf_nodes, round(mae_for_comparison),3), '\n')
    mae_for_comparison_list.append(mae_for_comparison)
    if mae_for_comparison == min(mae_for_comparison_list):
        optimal_max_leaf_nodes = max_leaf_nodes
print('Therefore, optimal number of leaf nodes is {} resulting in a MAE of {}. This is a {}% improvement over the MAE without adjusting the number of leaf nodes.\n'.format(optimal_max_leaf_nodes, round(mae_for_comparison),round((1-mae_for_comparison/mae_decision_tree_default)*100)))  

#Can we find a more optimal number of leaf nodes around the currently identified optimal number of leaf nodes?
range_bottom = int(optimal_max_leaf_nodes * 0.9)
range_top = int(optimal_max_leaf_nodes * 1.1) #must be int to be used in the rnage function. 
mae_for_comparison_from_refined_leaves_number_list = []
print('Testing of numbers of leaf nodes around initially identified optimal number of leaf nodes results in the following:\n')
for max_leaf_nodes in range(range_bottom, range_top):
    mae_for_comparison_from_refined_leaves_number = specify_fit_predict_validate(max_leaf_nodes, train_X, validation_X, train_y, validation_y)
    print('leaf nodes: {},    mae: {}'.format(max_leaf_nodes, round(mae_for_comparison_from_refined_leaves_number)), '\n')
    mae_for_comparison_from_refined_leaves_number_list.append(mae_for_comparison_from_refined_leaves_number)
    if mae_for_comparison_from_refined_leaves_number == min(mae_for_comparison_from_refined_leaves_number_list):
        refined_optimal_max_leaf_nodes = max_leaf_nodes
print('Therefore, optimal number of leaf nodes after further testing is {} resulting in a MAE of {}. This is a further {}% improvement over the MAE with the initial adjustment to the number of leaf nodes\n'.format(refined_optimal_max_leaf_nodes, round(min(mae_for_comparison_from_refined_leaves_number_list)), round((1-mae_for_comparison_from_refined_leaves_number/mae_for_comparison)*100)))        

#Will a different model perform better? Trying a random forest model.
#specify:
iowa_house_price_model_random_forest = RandomForestRegressor(random_state=1) #random forest performs quite well without tuning of paramaters. 
#fit
iowa_house_price_model_random_forest.fit(train_X, train_y)
#predict
predicted_house_prices_random_forest = iowa_house_price_model_random_forest.predict(validation_X)
#validate
mae_random_forest = mean_absolute_error(validation_y, predicted_house_prices_random_forest)   
print('Using a random forest model the MAE is {}. This is a {}% improvement over the MAE with more refined number of leaf nodes.\n\n'.format(round(mae_random_forest), round((1-mae_random_forest/mae_for_comparison_from_refined_leaves_number)*100)))
 
#STEP 8 - RE-SPECIFY AND FIT THE MODEL WITH IMPROVEMENTS BASED ON VALIDATION EXERCISE#

#Fit on all data (not just training data) to improve accuracy.
final_house_price_model = DecisionTreeRegressor(max_leaf_nodes=refined_optimal_max_leaf_nodes, random_state=1)
final_house_price_model.fit(X, y)







