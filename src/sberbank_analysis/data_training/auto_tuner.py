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
import multiprocessing as mp

class AutoTuner(object):
    """
    The purpose of this class is to auto-tune a given machine learning model to increase its performances.
    """

    def __init__(self, n_jobs = -1):
        """
        Class' constructor

        Parameters
        ----------
        n_jobs: integer (default = -1)
            Number of CPU cores to use for model training.

        Returns
        -------
        None
        """
        
        if n_jobs == -1:
            self.n_jobs = mp.cpu_count()
        else:
            self.n_jobs = n_jobs

    def _tune_model_helper(self, n_tries, model_class, hyperparameters_dict, X_train, y_train, X_test, y_test):
        """
        This method is a helper for model tuning.

        Parameters
        ----------
        n_tries : positive integer
                Number of different hyperparameter sets to try.

        model_class : Python class
                Class of the model that will be used as predictive model.

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
            lgb_params = {"objective": "regression",
                          "metric": "rmse",
                          "booster": "gbdt",
                          "num_leaves": int(params["num_leaves"]),
                          "max_depth": int(params["max_depth"]),
                          "learning_rate": params["learning_rate"],
                          "bagging_freq": int(params["bagging_freq"]),
                          "feature_fraction": params["feature_fraction"],
                          "bagging_fraction": params["bagging_fraction"],
                          "min_child_weight": params["min_child_weight"],
                          "min_split_gain": params["min_split_gain"],
                          "min_child_samples": int(params["min_child_samples"]),
                          "nthread": self.n_jobs}

            print("Params:", lgb_params)

            my_model = model_class(lgb_params)

            # Train the model
            my_model.fit(X_train, y_train)

            # Make predictions
            predictions_npa = my_model.predict(X_test)


            # Evaluate the model
            rmse = mean_squared_error(y_test, predictions_npa)**0.5
            
            print("RMSE = ", rmse)

            return rmse

        # Stores all information about each trial and about the best trial
        trials = Trials()

        best = fmin(fn=objective, trials=trials, space=hyperparameters_dict, algo=tpe.suggest, max_evals=n_tries)

        return best, trials

    def tune_model(self, n_tries, hyperparameters_dict, model_class, X_train, y_train, X_test, y_test):
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
        best, trials = self._tune_model_helper(n_tries, model_class, hyperparameters_dict, X_train, y_train, X_test, y_test)

        # Generate a DataFrame with results
        #results_df = pd.concat([results_df, pd.DataFrame(trials.vals, index=results_df.index)], axis=1)  # For LightGBM
        results_df = pd.DataFrame({"model": ["LGBM"] * len(trials.tids), "iteration": trials.tids,
                                   "RMSE": [x["loss"] for x in trials.results]})
        results_df = pd.concat([results_df, pd.DataFrame(trials.vals, index=results_df.index)], axis=1)  # For LightGBM
        results_df.sort_values("RMSE", ascending=True, inplace=True)
        return results_df