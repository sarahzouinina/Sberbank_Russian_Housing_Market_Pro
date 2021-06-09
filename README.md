[Kaggle BIPOC] Sberbank Russian Market Project
=====================================================

A regression analysis using LightGBM
======================================================

Project Description
------------
The project consists of analysing the data from the Sberbank Russian Housinng Market competition, check the link: https://www.kaggle.com/c/sberbank-russian-housing-market

The Sberbank Russian Market Project is the project I am working on for the BIPOC grant program under the supervision of Thomas Seleck (check his github here https://github.com/ThomasSELECK), At first did some very basic EDA, Feature Engineering and I used the LGBM model for prediction, the first attempt gave an rmse of 0.48122.
```python
params = {
        "objective" : "regression",
        "metric" : "rmse",
        "num_leaves" : 64,
        "learning_rate" : 0.01,
        'max_depth' : -1,
        'colsample_bytree': 0.9,
        'num_leaves': 150,
        "bagging_seed" : 42,
        "verbosity" : 1,
        "seed": 42,
        }
```
Second Attempt
In our case we are going to use LGBM with RMSE as an evaluation metric, withhout changing the model parameters and just by changing the median to mean the rmse dropped to 0.42226 which is a greate improvement regarding the simple change.

Tip: In case of MAE it's better to use the median :)

rmse = 0.40010
```python
params = {
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
    "seed": 42,
    'boosting_type': 'dart',
}

```
Final submission
I did dived deep into data and did a more elaborated EDA and created some new variables using feature engineering. Then I used hyperopt to look for the best model's parameters and applied my LGBM model. After many trials, hyperparameter tweakin and different preprocessing, the best rmse I obtained was 0.33017
```python
params =  {"metric": "rmse", 
           'bagging_fraction': 0.9537839478758712, 
           'bagging_freq': 4,
           'feature_fraction': 0.2022879151888015,
           'learning_rate': 0.01,
           'max_depth': 7, 
           'min_child_samples': 142,
           'min_child_weight': 0.0734986928127401, 
           'min_split_gain': 0.0613574767307937, 
           'num_leaves': 18}

```


Project Organization
------------

    ├── data                                     <- All the data of the project.
    ├── model_tuning_results                     <- Model's tuning results
    │
    ├── notebooks                                <- Jupyter notebooks. Drafts for data exploration and visualizations.
    ├── plots                                    <- Figures and graphics of the project.
    │
    ├── predictions                              <- The submission files
    │
    ├── src                                      <- Source code for use in this project.
    │   ├── dev                                  <- Contains the main files
    │   │   └── main.py
    │   │   └── main_too.py
    │   │
    │   ├── sberbank_analysis
    │   │   │   └── data_preprocessing           <- Source code for the preprocessing and feature engineering steps.
    │   │   │   │   └── preprocessing_steps.py
    │   │   │   └── data_training                <- Source code for loading the data, tuning the model and training it.    
    │   │   │   │   └── auto_tuner.py
    │   │   │   │   └── file_paths.py
    │   │   │   │   └── lgbm.py
    │   │   │   │   └── loading.py
    │   │   │   └── data_visualization            <- Source code for generating visualizations.  
    │   │   │   │   └── generate_plots.py
    │   │   │   └── feature_engineering           <- Source code for correlation and feature selection.  
    │   │   │   │   └── feature_selector.py
    │   │   │   └── models
    │   │   │   │   └── sberbank_model.py         <- Source code for the Sberbank model constructor.  

