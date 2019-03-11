# -*- coding: utf-8 -*-
"""
Created on Sun Mar 10 20:47:21 2019

@author: loren
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import Imputer

#Build, train and validate on Iowa data:
iowa_data_file_path = 'input/train.csv'
iowa_data = pd.read_csv(iowa_data_file_path) 
y = iowa_data.SalePrice
X = iowa_data.drop(['SalePrice'], axis=1)
X = X.select_dtypes(exclude=['object'])
list_of_numerical_columns_with_missing_values = [col for col in X if X[col].isnull().any()]
X_extn = X.copy()
for col in list_of_numerical_columns_with_missing_values:
    X_extn[col + '_had_missing_values'] = X_extn[col].isnull()
    X[col + '_had_missing_values'] = X[col].isnull()
column_headers_X_extn = list(X_extn)
train_X_extn, validate_X_extn, train_y, validate_y = train_test_split(X_extn, y, random_state=1)
imputer = Imputer()
X = pd.DataFrame(imputer.fit_transform(X))
X.columns = column_headers_X_extn
train_X_extn = pd.DataFrame(imputer.fit_transform(train_X_extn))
train_X_extn.columns = column_headers_X_extn
validate_X_extn = pd.DataFrame(imputer.transform(validate_X_extn))
validate_X_extn.columns = column_headers_X_extn
model = RandomForestRegressor(random_state=1)
model_fit = model.fit(train_X_extn, train_y)
model_predictions_training = model_fit.predict(validate_X_extn)
model_assessment_mae = mean_absolute_error(model_predictions_training, validate_y)
print('Mae: ', model_assessment_mae)
#Apply model to test data:
test_data_file_path = 'input/test.csv'
test_data = pd.read_csv(test_data_file_path) 
test_X = test_data.select_dtypes(exclude=['object'])
test_X_extn = test_X.copy()
for col in list_of_numerical_columns_with_missing_values:
    test_X_extn[col + '_had_missing_values'] = test_X_extn[col].isnull()
test_X_extn = imputer.fit_transform(test_X_extn)
test_X_extn = imputer.fit_transform(test_X_extn)
test_X_extn = pd.DataFrame(test_X_extn)
test_X_extn.columns = column_headers_X_extn
model_fit = model.fit(X, y)
model_predictions_final = model_fit.predict(test_X_extn)
#format for Kaggle competition submission:
output = pd.DataFrame({'Id': test_X_extn.Id,
                       'SalePrice': model_predictions_final})
output.to_csv('submission.csv', index=False)



