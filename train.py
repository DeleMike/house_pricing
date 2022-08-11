#!/usr/bin/env python
# coding: utf-8

# # House Median Value Pricing
# 
# Content
# This project has a dataset that contains the **median house prices** for California districts derived from the 1990 census.
# Hence, using this dataset, I will try to predict the ***median house value*** for a particular house given the features of the house

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import mutual_info_score
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import export_text
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
import xgboost as xgb
import pickle



# Loading and Reading Data
print('Loading data...')
df = pd.read_csv('housing.csv')
print('loaded data')

# Data cleaning:  Null value removal and string conversion
# fill all null values with 0 and convert ocean_proximity contents to lower strings
# convert all strings to lower case and remove spacing
string_columns = list(df.dtypes[df.dtypes == 'object'].index)
for col in string_columns:
    df[col] = df[col].str.lower().str.replace(' ', '_').replace('<1h_ocean', 'less_than_hundred_mile_to_ocean')
# fill empty values with 0
df = df.fillna(0)

median_house_val_logs =  np.log1p(df.median_house_value)
df['median_house_val_logs'] = median_house_val_logs
del df['median_house_value']

# split the categorical and numerical columns
categorical =list(df.select_dtypes('object').columns)
numerical = list(df.drop(['median_house_val_logs'], axis=1).select_dtypes('number').columns)
features =  numerical + categorical

# split data
df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=43)
df_train, df_val = train_test_split(df_full_train, test_size=0.25, random_state=43)
# sanity check
print('Splitted data set: ', (len(df_train) / len(df)), (len(df_val) / len(df)), (len(df_val) / len(df)))

df_train = df_train.reset_index(drop=True)
df_val = df_val.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)

y_train = df_train.median_house_val_logs.values
y_val = df_val.median_house_val_logs.values
y_test = df_test.median_house_val_logs.values

# drop target column
df_train = df_train.drop('median_house_val_logs', axis=1)
df_val = df_val.drop('median_house_val_logs', axis=1)
df_test = df_test.drop('median_house_val_logs', axis=1)

print('cleaned and prepared data...')

# TRAINING
print('Training started.')
#training dataset
train_dicts = df_train.to_dict(orient='records')
dv = DictVectorizer(sparse=False)
X_train = dv.fit_transform(train_dicts)

# validation dataset
val_dicts = df_val.to_dict(orient='records')
X_val = dv.transform(val_dicts)
features = dv.get_feature_names_out()

dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=features)
dval = xgb.DMatrix(X_val, label=y_val, feature_names=features)
# parameters
xgb_params = {
    'eta': 0.1, 
    'max_depth': 12,
    'min_child_weight': 1,
    
    'objective': 'reg:squarederror',
    'eval_metric': 'rmse',
    'nthread': 8,
    'seed': 1,
    'verbosity': 1,
}
model = xgb.train(xgb_params, dtrain, num_boost_round=100)

y_pred = model.predict(dval)
score = np.sqrt(mean_squared_error(y_val, y_pred))
print('rmse score on validation dataset is: ', score)

# TRAINING ON FULL_TRAIN DATASET
print('Training the final model...')
df_full_train = df_full_train.reset_index(drop=True)
y_full_train = df_full_train.median_house_val_logs

# remove target column
del df_full_train['median_house_val_logs']

dicts_full_train = df_full_train.to_dict(orient='records')
X_full_train = dv.fit_transform(dicts_full_train)

dicts_test = df_test.to_dict(orient='records')
X_test = dv.transform(dicts_test)

dfulltrain = xgb.DMatrix(X_full_train, label=y_full_train,
                    feature_names=dv.get_feature_names_out())
dtest = xgb.DMatrix(X_test, feature_names=dv.get_feature_names_out())

xgb_params = {
    'eta': 0.1, 
    'max_depth': 12,
    'min_child_weight': 1,
    
    'objective': 'reg:squarederror',
    'eval_metric': 'rmse',
    'nthread': 8,
    'seed': 1,
    'verbosity': 1,
}
model = xgb.train(xgb_params, dtrain, num_boost_round=100)

y_pred = model.predict(dtest)
score = np.sqrt(mean_squared_error(y_test, y_pred))
print('rmse score on the final model is: ', score)


# Save the model
# saving model
print('Saving model...')
output_file = 'model_v0.bin'

with open(output_file, 'wb') as f_out:
    pickle.dump((dv, model), f_out)

print('Model saved as: {}'.format(output_file))
