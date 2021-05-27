################################################################################
# Bristol-Myers Squibb Molecular Translation competition                       #
#                                                                              #
# This file contains a class defining an auto-tuning system.                   #
# Developed using Python 3.8.                                                 #
#                                                                              #
# Author: Thomas SELECK                                                        #
# e-mail: thomas.seleck@outlook.fr                                             #
# Date: 2021-03-07                                                             #
# Version: 1.0.0                                                               #
################################################################################

import numpy as np
import pandas as pd
from hyperopt import hp, fmin, tpe, Trials
from sklearn.metrics import mean_squared_error
import pickle
from src.sberbank_analysis.data_training.lgbm import LGBMRegressor


class AutoTuner(object):
    """
    The purpose of this class is to auto-tune a given machine learning model to increase its performances.
    """

    def __init__(self):
        """
        Class' constructor

        Parameters
        ----------
        None

        Returns
        -------
        None
        """

        pass

    def _tune_model_helper(self, n_tries, hyperparameters_dict, X_train, y_train):
        """
        This method is a helper for model tuning.

        Parameters
        ----------
        n_tries : positive integer
                Number of different hyperparameter sets to try.

        hyperparameters_dict : Python dictionary
                Dictionary containing hyperparameters to tune and corresponding range.

        X_train : Pandas DataFrame
                This is the data we will use to train the models.

        y_train : Pandas Series
                This is the associated target to X_train.

        X_test : Pandas DataFrame
                This is the data we will use to evaluate the models.

        y_test : Pandas Series
                This is the associated target to X_test.

        Returns
        -------
        None
        """

        def objective(params):
            lgb_params = {"num_leaves": hp.quniform("num_leaves", 8, 128, 2),
                          "learning_rate": hp.uniform("learning_rate", 1e-3, 1e-2),
                          "bagging_fraction": hp.uniform("bagging_fraction", 0.7, 1.0),
                          "feature_fraction": hp.uniform("feature_fraction", 0.1, 0.5),
                          "min_split_gain": hp.uniform("min_split_gain", 0.01, 0.1),
                          "min_child_samples": hp.quniform("min_child_samples", 90, 200, 2),
                          "min_child_weight": hp.uniform("min_child_weight", 0.01, 0.1)}

            lgbm = LGBMRegressor(lgb_params, early_stopping_rounds=150, test_size=0.25, verbose_eval=100,
                                 nrounds=5000, enable_cv=True)

            # Train the model
            lgbm.fit(X_train, y_train)

            # Make predictions
            predictions_npa = lgbm.predict(X_eval)

            # Evaluate the model
            rmse = mean_squared_error(y_eval, predictions_npa)**0.5
            print("RMSE = ", rmse)

            return rmse

        # Stores all information about each trial and about the best trial
        trials = Trials()

        best = fmin(fn=objective, trials=trials, space=hyperparameters_dict, algo=tpe.suggest, max_evals=n_tries)

        return best, trials

    def tune_model(self, n_tries, hyperparameters_dict, X_train, y_train):
        """
        This method tunes a ML model.

        Parameters
        ----------
        n_tries : positive integer
                Number of different hyperparameter sets to try.

        hyperparameters_dict : Python dictionary
                Dictionary containing hyperparameters to tune and corresponding range.

        model_class : Python class
                Class of the model that will be used as predictive model.

        X_train : Pandas DataFrame
                This is the data we will use to train the models.

        y_train : Pandas Series
                This is the associated target to X_train.

        X_test : Pandas DataFrame
                This is the data we will use to evaluate the models.

        y_test : Pandas Series
                This is the associated target to X_test.

        Returns
        -------
        None
        """

        # Preprocess the data
        # X_train = preprocessing_pipeline.fit_transform(X_train, y_train)
        # X_test = preprocessing_pipeline.transform(X_test)

        # Generate hyperparameters sets
        # self.hyperparameters_sets_lst = self._generate_random_hyperparameters_sets(hyperparameters_dict)

        # Actually do the tuning
        best, trials = self._tune_model_helper(n_tries, hyperparameters_dict, X_train, y_train)

        # Generate a DataFrame with results
        results_df = pd.concat([results_df, pd.DataFrame(trials.vals, index=results_df.index)], axis=1)  # For LightGBM


        results_df.sort_values("RMSE", ascending=True, inplace=True)

        return results_df