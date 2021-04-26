import pandas as pd
import numpy as np
np.set_printoptions(suppress=True)
from src.sberbank_analysis.data_preprocessing import preprocessing_steps
from src.sberbank_analysis.feature_engineering import feature_selector
from sklearn.preprocessing import StandardScaler
from src.sberbank_analysis.data_training import lgbm

################# Load Data ##############
pd.set_option('display.max_columns', None)
train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")
df_test = test
df_train = train

np.random.seed(0)
################## Preprocessing ######################

pr = preprocessing_steps.Preprocessor()
pr.fit(train)
train = pr.transform(train)
test = pr.transform(test)


print(train.head())
print(test.head())


################## Correlation ###########################

cc = feature_selector.Correlation_Checker()
train, test = cc.transform(train, test)
print(train.shape)
print(test.shape)
################## X_train, y_train ###############################
X_train = train.drop(['price_doc'], axis = 1)
y_train = train['price_doc']
X_test = test

sc = StandardScaler()
X_train, X_test = sc.fit_transform(X_train), sc.fit_transform(X_test)

################## Target Engineering ####################

y_train = np.log10(y_train)

################# Building the Model ####################

# RMSE = 0.40010
lgb_params = {
    "objective": "regression",
    "metric": "rmse",
    'learning_rate': 0.3777518392924809,
    'sub_feature': 0.5424987750103974,
    'max_depth': 94,
    'colsample_bytree': 0.9,
    'num_leaves': 194,
    "bagging_seed": 42,
    'min_data': 31,
    "verbosity": 1,
    "seed": 42
}

model = lgbm.LGBMRegressor(lgb_params, early_stopping_rounds = 150, test_size = 0.25, verbose_eval = 100, nrounds = 100, enable_cv = True)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
transformed_y_pred = 10 ** y_pred
# Submitting the file
my_submission = pd.DataFrame({'id': df_test.id, 'price_doc': transformed_y_pred})
# you could use any filename. We choose submission here
my_submission.to_csv('predictions/submission_100_rounds.csv', index=False) # With cross val RMSE = 0.40070