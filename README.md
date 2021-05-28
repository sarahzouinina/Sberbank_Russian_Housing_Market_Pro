# [Kaggle BIPOC] Sberbank Russian Market Project
The project consists of analysing the data from the Sberbank Russian Housinng Market competition, check the link: https://www.kaggle.com/c/sberbank-russian-housing-market

## Day1:
The Sberbank Russian Market Project is the first project I am working on under the BIPOC grant  program under the supervision of my Mentor Thomas Seleck (check his github here https://github.com/ThomasSELECK), I did some EDA, Feature Engineering and I used the LGBM model for prediction, the first attempt gave an rmse of 0.48122.
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
## Day2:
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


