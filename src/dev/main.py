import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score

np.set_printoptions(suppress=True)
from src.sberbank_analysis.data_preprocessing import preprocessing_steps
from src.sberbank_analysis.feature_engineering import feature_selector
from src.sberbank_analysis.data_training import loading
from src.sberbank_analysis.data_training import lgbm
from src.sberbank_analysis.data_training.files_paths import *


if __name__ == "__main__":

    ################# Load Data ##############
    ld = loading.Loader()
    train, test = ld.load_data(TRAINING_DATA_str, TESTING_DATA_str)
    df_test = test.copy()
    df_train = train.copy()
    ld.display_head(train)
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
    X_test = test.copy()

    ################## Target Engineering ####################

    y_train = np.log1p(y_train)

    ################# Building the Model ####################

    # RMSE = 0.40010
    lgb_params = {
        "objective": "regression",
        "metric": "rmse",
        'learning_rate': 0.001,
        'sub_feature': 0.5424987750103974,
        'max_depth': 94,
        'colsample_bytree': 0.9,
        'num_leaves': 194,
        "bagging_seed": 42,
        'min_data': 31,
        "verbosity": 1,
        "seed": 42
    }
    
    model = lgbm.LGBMRegressor(lgb_params, early_stopping_rounds = 150, test_size = 0.25, verbose_eval = 100, nrounds = 5000, enable_cv = True)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    transformed_y_pred = np.expm1(y_pred)
    # Submitting the file
    my_submission = pd.DataFrame({'id': df_test.id, 'price_doc': transformed_y_pred})
    # you could use any filename. We choose submission here
    my_submission.to_csv(r'C:\Users\SarahZOUININA\Documents\GitHub\Sberbank_Russian_Housing_Market_Pro\predictions\submission_5000_rounds_lr_0.01.csv', index=False) # With cross val RMSE = 0.32543