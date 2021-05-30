import pandas as pd
import numpy as np
np.set_printoptions(suppress=True)
from src.sberbank_analysis.models.sberbank_model import SberbankModel
from src.sberbank_analysis.feature_engineering import feature_selector
from src.sberbank_analysis.data_training import loading
from src.sberbank_analysis.data_training.file_paths import *

################# Load Data ##############
ld = loading.Loader()
train, test = ld.load_data(TRAINING_DATA_str, TESTING_DATA_str)
df_test = test.copy()
df_train = train.copy()
# ld.display_head(train)


################## Correlation ###########################

cc = feature_selector.Correlation_Checker()
train, test = cc.transform(train, test)
################## X_train, y_train ###############################
target = train['price_doc']
train = train.drop(['price_doc'], axis=1)
y_log_target = np.log1p(target)
results_df = pd.read_csv(TUNING_RESULTS_DIR_str + "\lgbm_tuning_results_no_ordinal.csv")
lgb_opt_dict = results_df.to_dict(orient='index')

lgb_dict = {"metric": "rmse",'bagging_fraction': 0.9364639030575194, 'bagging_freq': 4,
            'feature_fraction': 0.3983742888801865, 'learning_rate': 0.0010246921846085,
            'max_depth': 3, 'min_child_samples': 190, 'min_child_weight': 0.0935411651426567,
            'min_split_gain': 0.069319388501686, 'num_leaves': 18, 'subsample': 0.6322482970454842}
# no_ordinal = 0.34427


#lgb_dict = {"metric": "rmse", 'bagging_fraction': 0.9537839478758712, 'bagging_freq': 4,
#           'feature_fraction': 0.2022879151888015, 'learning_rate': 0.01, 'max_depth': 7, 'min_child_samples': 142,
#          'min_child_weight': 0.0734986928127401, 'min_split_gain': 0.0613574767307937, 'num_leaves': 18} #score_w_bagging = 0.33017

my_model = SberbankModel(lgb_dict)
my_model.fit(train, y_log_target)
predictions_npa = my_model.predict(test)
my_submission = pd.DataFrame({'id': df_test.id, 'price_doc': predictions_npa})
# # # # # you could use any filename. We choose submission here
my_submission.to_csv(PREDICTION_RESULTS_str + '\submission_opt_cleaned_no_ord.csv', index=False)