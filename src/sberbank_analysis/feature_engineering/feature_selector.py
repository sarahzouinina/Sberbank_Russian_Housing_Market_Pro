###############################################################################
# Sberbank Russian Housing Market Challenge                                   #
#                                                                             #
# This is the entry point of the solution.                                    #
# Developped using Python 3.8.                                                #
#                                                                             #
# Author: Sarah Zouinina                                                      #
# e-mail: sarahzouinina1@gmail.com                                            #
# Date: 2021-05-10                                                            #
# Version: 1.0.0                                                              #
###############################################################################


from sklearn.base import TransformerMixin, BaseEstimator

class Correlation_Checker(TransformerMixin, BaseEstimator):
    def __init__(self):
        """
        Parameters
        ----------
        None
        Attributes
        ----------
        None
        """

    def fit(self, X, y=None):
        """A reference implementation of a fitting function for a transformer.
        Parameters
        ----------
        X_train : {array-like, sparse matrix}, shape (n_samples, n_features)
            The training input samples.
        X_test : None
            There is no need of a target in a transformer, yet the pipeline API
            requires this parameter.
        treshold:
        Returns
        -------
        self : object
            Returns self.
        """
        return self

    def transform(self, X_train, X_test, treshold = 0.8):
        """ A reference implementation of a transform function.
        Parameters
        ----------
        X : {array-like, sparse-matrix}, shape (n_samples, n_features)
            The input samples.
        Returns
        -------
        X_transformed : array, shape (n_samples, n_features)
            The array containing the element-wise square roots of the values
            in ``X``.
        """
        col_corr_ = set()  # Set of all the names of correlated columns
        numerical_features = [feature for feature in X_train.columns if X_train[feature].dtype != 'O']
        corr_matrix_ = X_train[numerical_features].corr()
        for i in range(len(corr_matrix_.columns)):
            for j in range(i):
                if abs(corr_matrix_.iloc[i, j]) > treshold:
                    col_name_ = corr_matrix_.columns[i]  # getting the name of column
                    col_corr_.add(col_name_)

        X_train = X_train.drop(col_corr_, axis=1)
        X_test = X_test.drop(col_corr_, axis=1)

        return X_train, X_test

