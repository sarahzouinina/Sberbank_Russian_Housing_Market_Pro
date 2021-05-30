################################################################################
# Bristol-Myers Squibb Molecular Translation competition                       #
#                                                                              #
# This file contains a class defining a model based on a ResNet encoder and a  #
# LSTM decoder.                                                                #
# Developped using Python 3.8.                                                 #
#                                                                              #
# Author: Thomas SELECK                                                        #
# e-mail: thomas.seleck@outlook.fr                                             #
# Date: 2021-03-14                                                             #
# Version: 1.0.0                                                               #
################################################################################

import numpy as np
import pandas as pd
import time
import gc
import pickle

from src.sberbank_analysis.data_preprocessing.preprocessing_steps import Preprocessor
from src.sberbank_analysis.data_training.lgbm import LGBMRegressor

class SberbankModel(object):
    """
    This class defines a model for the Sberbank competition.
    """

    def __init__(self, lgb_params):
        self.pr = Preprocessor()
        self.lgbm = LGBMRegressor(lgb_params, early_stopping_rounds=150, test_size=0.25, verbose_eval=100,
                                  nrounds=5000, enable_cv=False)

    def fit(self, X_train, y_train):
        X_train = self.pr.fit_transform(X_train, y_train)

        self.lgbm.fit(X_train, y_train)

        return self

    def predict(self, X_test):
        X_test = self.pr.transform(X_test)

        predictions_npa = self.lgbm.predict(X_test)
        predictions_npa = np.expm1(predictions_npa)

        return predictions_npa