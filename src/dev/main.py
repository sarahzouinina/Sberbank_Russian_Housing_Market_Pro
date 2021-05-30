import pandas as pd
import numpy as np
import logging
import sys
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split

np.set_printoptions(suppress=True)
from src.sberbank_analysis.models.sberbank_model import SberbankModel
from src.sberbank_analysis.data_preprocessing import preprocessing_steps
from src.sberbank_analysis.feature_engineering import feature_selector
from src.sberbank_analysis.data_training import loading, auto_tuner
from src.sberbank_analysis.data_training import lgbm
from src.sberbank_analysis.data_training.file_paths import *
from custom_logging import CustomLogger # CHECK THAT PATH IS OK!
from datetime import datetime
from hyperopt import hp


if __name__ == "__main__":
    # Logger
    current_date = datetime.today().strftime("%d%m%Y")
    file_handler = logging.FileHandler(
        filename= "logs/" + "WholePipeline_LGBM_test_" + current_date + ".txt", mode="w",
        encoding="utf-8")
    sys.stdout = CustomLogger(sys.stdout, file_handler, "WholePipeline_LGBM_test.py")
    sys.stderr = CustomLogger(sys.stderr, file_handler, "stderr")

    ################# Load Data ##############
    ld = loading.Loader()
    train, test = ld.load_data(TRAINING_DATA_str, TESTING_DATA_str)
    df_test = test.copy()
    df_train = train.copy()
    #ld.display_head(train)


    ################## Correlation ###########################

    cc = feature_selector.Correlation_Checker()
    train, test = cc.transform(train, test)
    # print(train.shape)
    # print(test.shape)
    
    ################## X_train, y_train ###############################
    target = train['price_doc']
    train = train.drop(['price_doc'], axis = 1)
    y_log_target= np.log1p(target)
    
    X_train, X_test, y_train, y_test = train_test_split(train, y_log_target, test_size = 0.20, random_state = 42)

    ################# Tuning the Model ####################
    hyperparameters_dict = {"num_leaves": hp.quniform("num_leaves", 8, 128, 2),
                            "max_depth": hp.quniform("max_depth", 3, 10, 1),
                            "learning_rate": hp.uniform("learning_rate", 1e-3, 1e-2),
                            "bagging_freq": hp.quniform("bagging_freq", 1, 5, 1),
                            "bagging_fraction": hp.uniform("bagging_fraction", 0.7, 1.0),
                            "feature_fraction": hp.uniform("feature_fraction", 0.2, 0.9),
                            'subsample': hp.uniform('subsample', 0.6, 1),
                            "min_split_gain": hp.uniform("min_split_gain", 0.01, 0.1),
                            "min_child_samples": hp.quniform("min_child_samples", 90, 200, 2),
                            "min_child_weight": hp.uniform("min_child_weight", 0.01, 0.1)}

    at = auto_tuner.AutoTuner()
    results_df = at.tune_model(100, hyperparameters_dict, SberbankModel, X_train, y_train, X_test, y_test)
    results_df.to_csv(TUNING_RESULTS_DIR_str + "\lgbm_tuning_results_no_ordinal.csv", index=False)

