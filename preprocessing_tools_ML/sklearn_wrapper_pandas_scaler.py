# -*- coding: utf-8 -*-

import os
import pandas as pd
from sklearn.impute import SimpleImputer, MinMaxScaler, StandardScaler

################################################################################
# wrapper classes for sklearn to keep dataframes
# input: dataframe, output: dataframe
################################################################################



class PandasSimpleImputer(SimpleImputer):
    """A wrapper around `SimpleImputer` to return data frames with columns.
    """
    def fit(self, X, y=None):
        self.columns = X.columns
        return super().fit(X, y)

    def transform(self, X):
        return pd.DataFrame(super().transform(X), columns=self.columns)



class PandasStandardScaler(StandardScaler):
    """A wrapper around `SimpleImputer` to return data frames with columns.
    """
    def fit(self, X, y=None):
        self.columns = X.columns
        return super().fit(X, y)

    def transform(self, X):
        return pd.DataFrame(super().transform(X), columns=self.columns)


class PandasMinMaxScaler(MinMaxScaler):
    """A wrapper around `SimpleImputer` to return data frames with columns.
    """
    def fit(self, X, y=None):
        self.columns = X.columns
        return super().fit(X, y)

    def transform(self, X):
        return pd.DataFrame(super().transform(X), columns=self.columns)

#custom_scaler = PandasMinMaxScaler()
#PandasStandardScaler().fit_transform(X)
