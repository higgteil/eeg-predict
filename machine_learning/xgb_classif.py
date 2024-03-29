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




from imblearn import FunctionSampler
from imblearn.pipeline import make_pipeline

def outlier_rejection(X, y):
    """This will be our function used to resample our dataset."""
    model = IsolationForest(max_samples=1, contamination=0.1, random_state=1000)
    model.fit(X)
    y_pred = model.predict(X)
    return X[y_pred == 1], y[y_pred == 1]

reject_sampler = FunctionSampler(func=outlier_rejection)


import glob
direc= "/content/drive/MyDrive/data"
os.chdir(direc)
X = pd.read_csv(glob.glob("_*.csv")[0],encoding="utf-8", sep="\t", index_col=0)
y = X.groups_combined
X = X.drop(X.filter(regex="group|heavy|hs|ID").columns.to_list(),axis=1)

# make data for troubleshooting if needed
#from sklearn import datasets
#X, y = datasets.make_classification(
#    n_samples=110, n_features=15, n_informative=5, random_state=1
#)
#X=pd.DataFrame(X)
#y=pd.DataFrame(y)



# DIRECTORY
directory = "/content/drive/MyDrive/Smokers_Neversmokers_5foldCV"

tprs=[]

SEED = 1000
N_REPEATS = 5
# configure the cross-validation procedure
cv_outer = RepeatedStratifiedKFold(n_splits=10, n_repeats=N_REPEATS, random_state=SEED)
cv_inner = StratifiedKFold(n_splits=5)


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




# enumerate splits
outer_results = list()
# count
count = 0

conf_matrix_list_of_arrays=[]
labels=['smokers','never smokers']

# save // plot stuff
base_fpr = np.linspace(0, 1, 101)
fig, ax = plt.subplots()
tprs = []
aucs = []
mean_fpr = np.linspace(0, 1, 100)
results=[]

for train_ix, test_ix in cv_outer.split(X,y):
    # split data
    X_train, X_test = X.iloc[train_ix, :], X.iloc[test_ix, :]
    y_train, y_test = y.iloc[train_ix], y.iloc[test_ix]


    pipeline = imbpipeline(steps = [['preproc',preprocessor],
                                        ['smote',SMOTETomek()],
                                        ['outlier',reject_sampler],
                                        ['scaler', MinMaxScaler()],
                                        #['selectKBest',SelectKBest(f_classif,k=20)],
                                        ['classifier', XGBClassifier()]])
                                     
               
    
    param_grid={
                 ### xgb classifier
                'classifier__max_depth': [4,6,8],
                'classifier__learning_rate': [.1,1,2],
                'classifier__subsample': [.6,.7,.99],
                }   
    
    OptimPipe = GridSearchCV(pipeline, param_grid, scoring='accuracy', cv=cv_inner, n_jobs=-1)
    result = OptimPipe.fit(X_train, y_train)
    print('training cv_score:{}\n test_score:{}'.format(OptimPipe.best_score_, OptimPipe.score(X_test, y_test)))
    
   
    # get the best performing model fit on the whole training set
    best_model = OptimPipe.best_estimator_
    # evaluate model on the hold out dataset
    yhat = best_model.predict(X_test)
    ypred = best_model.predict_proba(X_test)

    # evaluate the model
    acc = accuracy_score(y_test, yhat)

    # Append
    results.append({'count':count,'yhat': 
                    yhat,'ypred':ypred,
                    'ytest':y_test,
                    'score_acc':[acc],
                    'training_score':OptimPipe.best_score_})
    
    y_pred = pd.DataFrame(ypred.argmax(axis=1))
    #conf_matrix = classification_report(list(y_test[0]),(list(y_pred[0])),target_names = labels, output_dict=True)
    conf_matrix = classification_report(list(y_test),list(ypred.argmax(axis=1)),target_names=labels,output_dict=True)

    # save classification_report for obtaining precision,recall,f1 and accuracy for later 
    conf_matrix_list_of_arrays.append({'count':count,
                                      'macro-avg':conf_matrix['macro avg'],
                                      'never smokers':conf_matrix['never smokers'],
                                      'smokers':conf_matrix['smokers'],
                                      'weighted_avg':conf_matrix['weighted avg'],
                                      'accuracy':conf_matrix['accuracy']})

    # store acc
    outer_results.append(acc)
    # report progress
    fpr, tpr, _ = roc_curve(y_test, ypred[:, 1])
    roc_auc = roc_auc_score(y_test, ypred[:, 1])

    interp_tpr = np.interp(mean_fpr, fpr, tpr)
    interp_tpr[0] = 0.0
    tprs.append(interp_tpr)
    aucs.append(roc_auc)
    count = count+1


ax.plot([0, 1], [0, 1], linestyle="--", lw=2, color="r", label="Chance", alpha=0.8)

mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
std_auc = np.std(aucs)
ax.plot(
    mean_fpr,
    mean_tpr,
    color="b",
    label=r"Mean ROC (AUC = %0.2f $\pm$ %0.2f)" % (mean_auc, std_auc),
    lw=2,
    alpha=0.8,
)

std_tpr = np.std(tprs, axis=0)
tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
tprs_lower = np.maximum(mean_tpr - std_tpr, 0)

tn = "ROC classification {} vs {}".format(labels[0],labels[1])


ax.fill_between(
    mean_fpr,
    tprs_lower,
    tprs_upper,
    color="lightcoral",
    alpha=0.2,
    label=r"$\pm$ 1 std. dev.",
)

ax.set(
    xlim=[-0.05, 1.05],
    ylim=[-0.05, 1.05],
    title=tn,
)
ax.legend(loc="lower right")


if not os.path.exists(directory):
   os.makedirs(directory)
os.chdir(directory)


#plt.tight_layout()
plt.savefig("ROC.png",dpi=400)
#plt.show()
plt.close()


    

# save

filename = 'grid_searched_xgb_model.sav'
model = OptimPipe.best_estimator_.steps[-1][1]
joblib.dump(model, filename)
