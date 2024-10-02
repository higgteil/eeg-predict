# -*- coding: utf-8 -*-

import os
import pandas as pd
import numpy as np 
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_validate
from sklearn.metrics import auc, roc_auc_score
import shap
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectPercentile, chi2,SelectFdr,SelectFwe
from sklearn.svm import SVC
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression

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
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import RidgeClassifier
from imblearn.ensemble import BalancedRandomForestClassifier
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
from sklearn.impute import SimpleImputer, KNNImputer
import warnings 
warnings.filterwarnings("ignore")

import matplotlib
#matplotlib.use('Agg')




import glob
direc= "/content/drive/MyDrive/data"
os.chdir(direc)
X = pd.read_csv(glob.glob("_*.csv")[0],encoding="utf-8", sep="\t", index_col=0)
y = X.groups_combined
X = X.drop(X.filter(regex="group|heavy|hs|ID").columns.to_list(),axis=1)


# DIRECTORY
directory = "/content/drive/MyDrive/SHAP"



# feature imputation (categorical/numerical)
features_categoric=["b05_soz_03","b05_soz_04","b05_soz_06","b05_soz_09","b05_soz_10","b05_soz_11","b05_soz_12","b05_soz_14","gender",
                        "mutter_001","mutter_002","vater_001","vater_002","vater_003","geschw_001","geschw_002","geschw_003",
                        "umfeld_001","mutter_005","mutter_004","c_haendig",
                        "umfeld_002","umfeld_003","umfeld_004","familie_001","familie_005","familie_004","familie_006","phase1_001","a_smoking","d_psychiatrisch"]

categorical_features = [col for col in X if col in features_categoric]
numeric_features = [col for col in X if col not in cols_cat]   

numeric_transformer = Pipeline(steps=[("imputer", SimpleImputer(strategy="mean"))])
categorical_transformer = Pipeline(steps=[("imputer", SimpleImputer(strategy="most_frequent"))])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)])







from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test= train_test_split(X_, y, test_size=0.33, random_state=1511)
cols = X.columns

impute = SimpleImputer(strategy="most_frequent")
X_train[cols_cat_] = impute.fit_transform(X_train[cols_cat_])
X_test[cols_cat_] = impute.fit_transform(X_test[cols_cat_])
impute = SimpleImputer(strategy="mean")
X_train[cols_num_] = impute.fit_transform(X_train[cols_num_])
X_test[cols_num_] = impute.fit_transform(X_test[cols_num_])

scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# back to dataframe
X_train = pd.DataFrame(X_train, columns=cols)
X_test = pd.DataFrame(X_test, columns=cols)

# predict
model = XGBClassifier()
model.fit(X_train ,y_train)
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)


sizes = [5,10]

for n in sizes:
    fig, ax = plt.subplots(figsize=(10,10),dpi=400) 
    #fig = shap.summary_plot(shap_values, X_test_p, plot_type="bar",max_display=5, show=False)
    fig = shap.summary_plot(shap_values, X_test, max_display=n, show=False)
    ax.tick_params(axis='y', labelsize=14)
    ax.tick_params(axis='x', labelsize=14)
#    plt.xlabel('average impact on model output magnitude \n mean (|SHAP|) value ', fontsize=16)
    plt.xlabel('SHAP value (impact on model output) ', fontsize=16)

    title ="SHAP summary of "+str(n)+" most important features"
    plt.title(title, fontsize=16)

    directory = "/content/drive/MyDrive/try/plots/SHAP_Heavy_vs_Light_Smokers/"
    import os
    if not os.path.exists(directory):
        os.makedirs(directory)
    else: os.chdir(directory)

    title="SHAP_HS_LS_bee"+str(n)+".png"
    plt.savefig(title,bbox_inches="tight")

    plt.show()
    plt.close()


# save SHAP values
filename = 'shap_values.pkl'
with open("shap_values.pkl","wb") as fp:
     pickle.dump(shap_values,fp)
