from sklearn.base import TransformerMixin, BaseEstimator
import pandas as pd
import numpy as np
pd.options.mode.chained_assignment = None  # default='warn'
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from sklearn.decomposition import PCA


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

        # Get the binary variables
        binary_features = ['product_type', 'culture_objects_top_25', 'thermal_power_plant_raion', 'incineration_raion',
                           'oil_chemistry_raion', 'radiation_raion', 'railroad_terminal_raion', 'big_market_raion',
                           'nuclear_reactor_raion', 'detention_facility_raion', 'water_1line', 'big_road1_1line',
                           'railroad_1line']

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
        X['sold_year'] = X['timestamp_parsed'].dt.year
        X['sold_month'] = X['timestamp_parsed'].dt.month
        X['sold_day'] = X['timestamp_parsed'].dt.day

        X = X.drop(['timestamp', 'timestamp_parsed'], axis=1)

        categorical_features.remove('timestamp')
        # categorical_features = list(set(categorical_features) - set(binary_features))

        def RemoveOutliers(df):
            df["build_year"].loc[df["build_year"] == 215] = 2015
            df["build_year"].loc[df["build_year"] == 71] = 1971
            df["build_year"].loc[df["build_year"] == 4965] = 1965
            df["build_year"].loc[df["build_year"] == 20] = 2000
            df["build_year"].loc[df["build_year"] == 20052009] = 2009
            df["build_year"].loc[(df["build_year"] > 3000) | (df["build_year"] < 1700)] = np.nan
            df['full_sq'].loc[df['full_sq'] == 5326] = 532.6
            df['full_sq'].loc[df['full_sq'] == 1] = 0
            df['life_sq'].loc[df['life_sq'] == 7478] = 747.8
            df["full_sq"].loc[df["full_sq"] < df["life_sq"]] = df["life_sq"].loc[df["full_sq"] < df["life_sq"]]
            df['kitch_sq'] = df['kitch_sq'].replace([620., 1970., 1974., 2013., 2014.],
                                                    [62.0, 197.0, 197.4, 201.3, 201.4])
            df['material'].iloc[df['material'] == 3] = 6
            return df

        X = RemoveOutliers(X)

        def create_new_features(df):
            bin_interval = [0, 1920, 1945, 1970, 1995, 2020]
            bin_labels = [1, 2, 3, 4, 5]
            df["living_area"] = df['full_sq'] - df['life_sq']
            df["percentage_living_area"] = df['life_sq'] / df['full_sq']
            df["room_area_average"] = df["living_area"] / df["num_room"]
            df["age_of_property"] = df["sold_year"] - df["build_year"]
            df['binned_build_year'] = pd.cut(X['build_year'], bins=bin_interval, labels=bin_labels)
            # how many people in m?
            df['area_by_person'] = df['area_m'] / df['raion_popul']
            # get the green area in the neighborhood
            df['green_area_m'] = df['green_zone_part'] * df['area_m']
            return df

        X = create_new_features(X)


        def DimensionalityReduction(df):
            # Reduce dimensionality for coffee shops related features
            coffeeRelatedColumns = df.filter(regex="cafe.*").columns
            pca = PCA(n_components=7)
            tmp = pca.fit_transform(df[coffeeRelatedColumns])
            df.drop(coffeeRelatedColumns, axis=1, inplace=True)
            df = pd.concat([df, pd.DataFrame(tmp, index=df.index, columns=["coffee_PCA_" + str(c) for c in
                                                                              range(tmp.shape[1])])], axis=1)

            #

            # Reduce dimensionality for religion related features
            religionRelatedColumns = df.filter(regex="(church.*)|(mosque.*)").columns.tolist()
            # Reduce dimensionality for people related features
            # peopleRelatedColumns = ["young_all", "young_male", "young_female", "work_all", "work_male", "ekder_all",
            #                         "ekder_female", "0_6_male", "0_6_female", "7_14_male", "7_14_female", "0_17_male",
            #                         "0_17_female"]
            # pca = PCA(n_components=2)
            # tmp = pca.fit_transform(df[peopleRelatedColumns])
            # df.drop(peopleRelatedColumns, axis=1, inplace=True)
            # df = pd.concat([df, pd.DataFrame(tmp, index=df.index, columns=["people_PCA_" + str(c) for c in
            #                                                                   range(tmp.shape[1])])], axis=1)
            pca = PCA(n_components=8)
            tmp = pca.fit_transform(df[religionRelatedColumns])
            df.drop(religionRelatedColumns, axis=1, inplace=True)
            df = pd.concat([df, pd.DataFrame(tmp, index=df.index, columns=["religion_PCA_" + str(c) for c in
                                                                              range(tmp.shape[1])])], axis=1)

            return df

        #X = DimensionalityReduction(X)


        for feature in X[categorical_features].columns:
            lb = LabelEncoder()
            X[feature] = lb.fit_transform(X[feature])

        for feature in X[binary_features].columns:
            oe = OrdinalEncoder()
            X[feature] = oe.fit_transform(X[[feature]])

        print("Preprocessing data... done")
        return X





