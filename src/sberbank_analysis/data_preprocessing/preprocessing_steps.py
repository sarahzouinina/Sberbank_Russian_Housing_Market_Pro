from sklearn.base import TransformerMixin, BaseEstimator
import pandas as pd
from sklearn.preprocessing import LabelEncoder


class Preprocessor(TransformerMixin, BaseEstimator):
    def __init__(self):
        """
        This is the class' constructor.
        Parameters
        ----------
        None
        Returns
        -------
        None
        """


    def fit(self, X,y=None):
        """
        This method is called to fit the transformer on the training data.
        Parameters
        ----------
        X : pd.Series
        This is a data frame containing the data used to fit the transformer.
        y : pd.Series (default = None)
        This is the target associated with the X data.
        Returns
        -------
        None
        """
        # Get features with NA
        # self.features_with_na = [feature for feature in X.columns if X[feature].isnull().sum() > 1]
        #
        # # Get categorical features
        # self.categorical_features = [feature for feature in X.columns if X[feature].dtype == 'O']
        #
        # # Get numerical features
        # self.numerical_features = [feature for feature in X.columns if X[feature].dtype != 'O']
        return self

    def transform(self, X):
        print("Preprocessing data...")
        # Get features with NA
        features_with_na = [feature for feature in X.columns if X[feature].isnull().sum() > 1]

        # Get categorical features
        categorical_features = [feature for feature in X.columns if X[feature].dtype == 'O']

        # Get numerical features
        numerical_features = [feature for feature in X.columns if X[feature].dtype != 'O']

        # Replace missing with a value
        X[categorical_features] = X[categorical_features].apply(lambda x: x.fillna('Missing'),axis=0)

        # Replacing ordinal features with the median
        ordinal_features = ['max_floor', 'material', 'num_room', 'build_year', 'state']
        X[ordinal_features] = X[ordinal_features].apply(lambda x: x.fillna(x.median()), axis=0)
        features_to_remove = ['id','price_doc', 'max_floor', 'material', 'num_room', 'build_year', 'state']
        numerical_features = list(set(numerical_features) - set(features_to_remove ))


        #Replacing numerical features with the mean
        X[numerical_features] = X[numerical_features].apply(lambda x: x.fillna(x.mean()),axis=0)

        #Dealing with time data
        X['timestamp_parsed'] = pd.to_datetime(X['timestamp'], format='%Y-%m-%d')  # Format 2011-08-20 : %Y-%m-%d
        X['year'] = X['timestamp_parsed'].dt.year
        X['month'] = X['timestamp_parsed'].dt.month
        X['day'] = X['timestamp_parsed'].dt.day

        X = X.drop(['timestamp', 'timestamp_parsed'], axis=1)

        categorical_features.remove('timestamp')

        for feature in X[categorical_features].columns:
            lb = LabelEncoder()
            X[feature] = lb.fit_transform(X[feature])

        print("Preprocessing data... done")
        return X





