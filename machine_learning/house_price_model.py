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
from sklearn.model_selection import cross_val_score

            ##########Notes/To Dos:##########
#show how choices were made about building the model e.g.:
    #choosing random forest over decision tree
    #comparing MAE for missing value handling (dropping, imputing, imputing while tracking what was mising)
            ##########End of Notes/To Dos:##########
            
#Build, train and validate on Iowa data:
iowa_data_file_path = 'input/train.csv'
iowa_data = pd.read_csv(iowa_data_file_path) 
y = iowa_data.SalePrice
X_with_objects = iowa_data.drop(['SalePrice'], axis=1)
X = X_with_objects.select_dtypes(exclude=['object'])
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
print('MAE - RF, IMPUTE AND TRACKING, NO OBJECTS: ', model_assessment_mae)

#Test impact of including categoricals:
X_with_objects_ohencoded = pd.get_dummies(X_with_objects)
list_of_object_and_num_columns_with_missing_values = [col for col in X_with_objects_ohencoded if X_with_objects_ohencoded[col].isnull().any()]
for col in list_of_object_and_num_columns_with_missing_values:
    X_with_objects_ohencoded[col + '_had_missing_values'] = X_with_objects_ohencoded[col].isnull()
column_headers_X_with_objects_ohencoded = list(X_with_objects_ohencoded)
train_X_with_obj_ohe, validate_X_with_obj_ohe, train_y, validate_y = train_test_split(X_with_objects_ohencoded, y, random_state=1)
imputer = Imputer()
train_X_with_obj_ohe = pd.DataFrame(imputer.fit_transform(train_X_with_obj_ohe))
train_X_with_obj_ohe.columns = column_headers_X_with_objects_ohencoded
validate_X_with_obj_ohe = pd.DataFrame(imputer.transform(validate_X_with_obj_ohe))
validate_X_with_obj_ohe.columns = column_headers_X_with_objects_ohencoded
model_ohe = RandomForestRegressor(random_state=1)
model_fit_ohe = model_ohe.fit(train_X_with_obj_ohe, train_y)
model_predictions_training_ohe = model_fit_ohe.predict(validate_X_with_obj_ohe)
model_assessment_mae_ohe = mean_absolute_error(model_predictions_training_ohe, validate_y)
print('Mae - RF, IMPUTE AND TRACKING, ONE-HOT ENCODING OBJECTS: ', model_assessment_mae)

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
output = pd.DataFrame({'Id': test_X_extn.Id.astype(int),
                       'SalePrice': model_predictions_final})
print(output.dtypes)
#output.Id = output.Id.astype(float)
output.to_csv('submission.csv', index=False)



