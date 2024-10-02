# -*- coding: utf-8 -*-


import os
import pandas as pd
import numpy as np 
from sklearn.datasets import make_classification

from sklearn.model_selection import cross_validate
from sklearn.metrics import auc, roc_auc_score
from sklearn.datasets import make_classification
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectPercentile, chi2,SelectFdr,SelectFwe
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.svm import SVC 
from sklearn import tree
import xgboost 
from xgboost import XGBClassifier
from imblearn.combine import SMOTEENN 
from imblearn.over_sampling import ADASYN, SVMSMOTE, BorderlineSMOTE,KMeansSMOTE
from sklearn.metrics import precision_score, recall_score,f1_score, balanced_accuracy_score
from sklearn.ensemble import StackingClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF

from sklearn.linear_model import RidgeClassifier
from sklearn.svm import LinearSVC
from sklearn.utils.fixes import loguniform
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import RandomizedSearchCV
import matplotlib.pylab as plt 
from sklearn.metrics import accuracy_score
from imblearn.combine import SMOTETomek
from imblearn.under_sampling import TomekLinks
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as imbpipeline
from sklearn.pipeline import Pipeline
from numpy import mean
from numpy import std
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold

####################################################################################################
# mvgroup imputer: function that does groupwise imputation in a cross-validation scheme, depending
# on the input feature (categorical or numerical). 
# Also, you need to include your target variable (target vector), otherwise it can't to the groupwise
# imputation. Although it is possible to code it with y as input, it may lead to problems because
# removing samples doesn't comply with sklearn standards, see e.g. here:
# source -> stackoverflow.com/questions/25539311/custom-transformer-for-sklearn-pipeline-that-alters-both-x-and-y
# INPUT: only works with dataframe


# QuantileOutlierRemover
# I modified the original class, so that categorical features are ignored. 
# Thanks a lot for this nice little class:
# source -> towardsdatascience.com/creating-custom-transformers-using-scikit-learn-5f9db7d7fdb5
# INPUT: only works with dataframe

####################################################################################################

import warnings 
warnings.filterwarnings("ignore")

import matplotlib
#matplotlib.use('Agg')


# Class for Quantile Outlier Removal (IQR)
class QuantileOutlierRemover(BaseEstimator,TransformerMixin):
    def __init__(self,factor=1.5):
        self.factor = factor
        # factor 1.5 (often used)
        # factor 3 (extreme outliers)
    def outlier_detector(self,X,y=None):
        X = pd.Series(X).copy()
        q1 = X.quantile(0.25)
        q3 = X.quantile(0.75)
        iqr = q3 - q1
        self.lower_bound.append(q1 - (self.factor * iqr))
        self.upper_bound.append(q3 + (self.factor * iqr))

    def fit(self,X,y=None):
        self.lower_bound = []
        self.upper_bound = []
        X.apply(self.outlier_detector)
        return self
    
    def transform(self,X,y=None):
        X = pd.DataFrame(X).copy()
        cols_to_find = ["gender","d_psychiatrisch","vater_001","vater_002","vater_003","mutter_001","mutter_002","mutter_003","mutter_004",
                        "c_haendig","geschw_001","geschw_002","geschw_003","umfeld_001","umfeld_002","umfeld_003","umfeld_004",
                        "familie_001","familie_004","familie_005","familie_006","geschw_003",
                        "b05_soz_03","b05_soz_04","b05_soz_06","b05_soz_08","b05_soz_10","b05_soz_10a","b05_soz_11",
                        "b05_bdi20","a_smoking","z_day","z_lifetime","phase1_001",
                        "b05_soz_14","mutter_pregnancy_smoking","groups_combined"]
                        
        cols2ignore = [X.columns.get_loc(col) for col in X if col in cols_to_find]
        cols2ignore.sort()
        for i in range(X.shape[1]):
            if i in cols2ignore:
               continue
            else:
                x = X.iloc[:, i].copy()
                x[(x < self.lower_bound[i]) | (x > self.upper_bound[i])] = np.nan
                X.iloc[:, i] = x
        return X






# function for sklearn multivariate groupwise imputation
def mvgroup_imputer(X,y=None): 
   #ignore = ["d_psychiatrisch"]
    imp = IterativeImputer(max_iter=10, random_state=0)

    features_categoric=["b05_soz_03","b05_soz_04","b05_soz_06","b05_soz_09","b05_soz_10","b05_soz_11","b05_soz_12","b05_soz_14","gender",
                        "mutter_001","mutter_002","vater_001","vater_002","vater_003","geschw_001","geschw_002","geschw_003",
                        "umfeld_001","mutter_005","mutter_004","umfeld_002","umfeld_003","umfeld_004","familie_001","familie_005",
                        "familie_004","familie_006","phase1_001","a_smoking","d_psychiatrisch"]

    cols_cat = [col for col in X if col in features_categoric or col =="groups_combined"]
    cols_num = [col for col in X if col not in cols_cat or col=="groups_combined"]
    cols_cat_ = [col for col in X if col in features_categoric]
    cols_num_ = [col for col in X if col not in cols_cat]   

    X[cols_cat_]=X.groupby("groups_combined")[cols_cat_].transform(lambda x: x.fillna(x.value_counts().index[0]))    
    X[cols_num_] = X.groupby("groups_combined")[cols_num_].transform(lambda col:imp.fit_transform(col.to_frame()).flatten())
    X = X.drop("groups_combined",axis=1)

    return X


# Isolation Forest Outlier Detection
from imblearn import FunctionSampler
from imblearn.pipeline import make_pipeline

def outlier_rejection(X, y):
    """This will be our function used to resample our dataset."""
    model = IsolationForest(max_samples=100, contamination="auto", random_state=1000)
    model.fit(X)
    y_pred = model.predict(X)
    return X[y_pred == 1], y[y_pred == 1]

reject_sampler = FunctionSampler(func=outlier_rejection)



# preprocessing (Sklearn multivariate & groupwise imputer for use in sklearn pipeline)
preproc_mvgi = FunctionTransformer(mvgroup_imputer)


# configure the cross-validation procedure
SEED = 1000
N_REPEATS = 1
cv_outer = RepeatedStratifiedKFold(n_splits=10, n_repeats=N_REPEATS, random_state=SEED)
cv_inner = StratifiedKFold(n_splits=4)

# X,y must be of type DataFrame!
for train_ix, test_ix in cv_outer.split(X,y):
    # split data
    X_train, X_test = X.iloc[train_ix, :], X.iloc[test_ix, :]
    y_train, y_test = y.iloc[train_ix], y.iloc[test_ix]




    pipeline = imbpipeline(steps = [['impute',preproc_mvgi],
                                        ['smote',SMOTETomek()],
                                        ['scaler', MinMaxScaler()],
                                        ['outlier',reject_sampler],
                                        #['selectKBest',SelectKBest(f_classif,k=20)],
                                        ['classifier', XGBClassifier()]])

                                     
    
    param_grid={
                 ### xgb classifier
                'classifier__max_depth': [4, 6, 8],
               # 'classifier__subsample': [0.6,0.9],
               # 'classifier__gamma': [.001,0.1,0.5,1]
                }                
    

    
    OptimPipe = RandomizedSearchCV(pipeline, param_grid, scoring='accuracy', cv=cv_inner, n_jobs=-1)
    OptimPipe.fit(X_train, y_train)
    print("\nBest parameter (CV score=%0.3f):" % OptimPipe.best_score_)
    print("model (test) score: %.3f" % pipeline.score(X_test, y_test))
