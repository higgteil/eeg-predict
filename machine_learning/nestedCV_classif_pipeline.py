#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 19 19:50:05 2022

@author: pablo
"""


#SEED = 1000
#model = XGBClassifier()
#cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=10, random_state=1)
#pipeline = imbpipeline(steps = [['smote',SMOTETomek()],['classifier', model]])
#score_nested = cross_validate(pipeline, X, y, cv=cv, scoring = 'accuracy', return_estimator =True)





import os
import pandas as pd
import numpy as np 
import pickle
import shap
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator,TransformerMixin
from sklearn.preprocessing import FunctionTransformer


from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler



from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectPercentile, chi2,SelectFdr,SelectFwe
from sklearn.svm import SVC
from sklearn.feature_selection import SelectFromModel
from sklearn import tree
import xgboost 
from xgboost import XGBClassifier
from imblearn.over_sampling import ADASYN, SVMSMOTE, BorderlineSMOTE,KMeansSMOTE
from sklearn.metrics import precision_score, recall_score,f1_score, balanced_accuracy_score, auc
from sklearn.ensemble import StackingClassifier, IsolationForest
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import RandomizedSearchCV
import matplotlib.pylab as plt 
from sklearn.metrics import accuracy_score
from imblearn.combine import SMOTETomek
from imblearn.under_sampling import TomekLinks
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, StratifiedKFold, train_test_split
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as imbpipeline
from sklearn.pipeline import Pipeline
from numpy import mean
from numpy import std
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
from sklearn.metrics import classification_report

from sklearn.preprocessing import StandardScaler, MinMaxScaler


# q50 notna
import glob
direc= "/fast/work/users/reinharp_c/projects/eeg_nestedCV/HeavySmokers_vs_Neversmokers/"
os.chdir(direc)
#X = pd.read_csv(glob.glob("X_*hs_renamed3.csv")[0],encoding="utf-8", sep="\t", index_col=0)
X = pd.read_csv(glob.glob("X_*xmas.csv")[0],encoding="utf-8", sep="\t", index_col=0)
X = X.reset_index(drop=True)
y = X.heavy_smokers
X = X.drop(X.filter(regex="group|heavy|hs|ID").columns.to_list(),axis=1)










import shap
from sklearn import tree
import xgboost 
from xgboost import XGBClassifier





import warnings 
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use('Agg')

#########################################################################################################################################################################################
features_categoric=["b05_soz_03","b05_soz_04","b05_soz_06","b05_soz_09","b05_soz_10","b05_soz_11","b05_soz_12","b05_soz_14","gender",
                        "mutter_001","mutter_002","vater_001","vater_002","vater_003","geschw_001","geschw_002","geschw_003",
                        "umfeld_001","mutter_005","mutter_004","c_haendig",
                        "umfeld_002","umfeld_003","umfeld_004","familie_001","familie_005","familie_004","familie_006","phase1_001","a_smoking","d_psychiatrisch"]

categorical_features = [col for col in X if col in features_categoric]
numeric_features = [col for col in X if col not in categorical_features]   


numeric_transformer = Pipeline(steps=[("imputer", SimpleImputer(strategy="mean"))])
categorical_transformer = Pipeline(steps=[("imputer", SimpleImputer(strategy="most_frequent"))])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)])

#########################################################################################################################################################################################





from datetime import datetime
start_time = datetime.now()



from imblearn import FunctionSampler
from imblearn.pipeline import make_pipeline

def outlier_rejection(X, y):
    """This will be our function used to resample our dataset."""
    model = IsolationForest(max_samples=1, contamination="auto", random_state=1510)
    model.fit(X)
    y_pred = model.predict(X)
    return X[y_pred == 1], y[y_pred == 1]

reject_sampler = FunctionSampler(func=outlier_rejection)



from sklearn.model_selection import KFold


# Reproducibility
np.random.seed(1)  

# repetitions 
N_CV_REPEATS = 5




# DIRECTORY
import time
timestr = time.strftime("%Y-%m-%d-%H-%M-%S")

fname = "nestedCV_"+str(N_CV_REPEATS)+"simple_imputer_new_repeats_"+timestr
directory = os.path.join(direc,fname)


if not os.path.exists(directory):
       os.makedirs(directory)
else: os.chdir(directory)









# Make a list of random integers between 0 and 10000 of length = N_CV_repeats to act as different data splits
random_states = np.random.randint(10000, size=N_CV_REPEATS) 

######## Use a dict to track the SHAP values of each observation per CV repitition 

shap_values_per_cv = dict()
for sample in X.index:
    ## Create keys for each sample
    shap_values_per_cv[sample] = {} 
    ## Then, keys for each CV fold within each sample
    for CV_repeat in range(N_CV_REPEATS):
        shap_values_per_cv[sample][CV_repeat] = {}




tprs=[]


SEED = 1510


# enumerate splits
outer_results = list()
# count
count = 0

conf_matrix_list_of_arrays=[]
labels=['heavy smokers','never smokers']

# save // plot stuff
base_fpr = np.linspace(0, 1, 101)
fig, ax = plt.subplots()
tprs = []
aucs = []
mean_fpr = np.linspace(0, 1, 100)
results=[]


from sklearn.model_selection import KFold, StratifiedKFold

for i, CV_repeat in enumerate(range(N_CV_REPEATS)): 

    #Establish CV scheme
    CV = StratifiedKFold(n_splits=10, shuffle=True, random_state=random_states[i]) # Set random state 
    

    ix_training, ix_test = [], []
    # Loop through each fold and append the training & test indices to the empty lists above
    for fold in CV.split(X,y):
        ix_training.append(fold[0]), ix_test.append(fold[1])
        
    ## Loop through each outer fold and extract SHAP values 
    for i, (train_outer_ix, test_outer_ix) in enumerate(zip(ix_training, ix_test)): 
        #Verbose
        print('\n------ Fold Number:',i)
        X_train, X_test = X.iloc[train_outer_ix, :], X.iloc[test_outer_ix, :]
        y_train, y_test = y.iloc[train_outer_ix], y.iloc[test_outer_ix]


        ## Establish inner CV for parameter optimization #-#-#
        cv_inner = StratifiedKFold(n_splits=4, shuffle=True, random_state=1510)

        pipeline = imbpipeline(steps = [
                                    #['preproc',preprocessor],
                                    ['preproc',SimpleImputer(strategy="most_frequent")],
                                    ['smote',SMOTETomek()],
                                    ['outlier',reject_sampler],
                                    ['scaler', MinMaxScaler()],
                                    #['selectKBest',SelectKBest(f_classif,k=20)],
                                    ['classifier', XGBClassifier()]])
        
        # A parameter grid for XGBoost
       # params = {
       #         'classifier__max_depth':[6, 8, 10, 12],
       #         'classifier__subsample': [.6,.7, 1.0],
       #         'classifier__colsample_bytree': [.6, .8, 1.0]}
       #         #'classifier__gamma': [0,.1,.5]}
                
                 
                
                
                
        #params = {'classifier__max_depth': [4, 6, 8],
        #         'classifier__learning_rate': [.01,.1],
        #         'classifier__subsample': [.7,1.0]}
        
                        
        params = {
                  'classifier__max_depth':[4, 6, 8, 10],
                  'classifier__learning_rate': [.01,.1,.2,.3],
                  'classifier__subsample': [.7, .8, .9,1]
                 }
                
                
                
        # Search to optimize hyperparameters
        search = RandomizedSearchCV(pipeline, params, cv=cv_inner) #-#-#
        search.fit(X_train, y_train) #-#=#
        #model.fit(X_train, y_train)

        # evaluate the model
        yhat = search.predict(X_test)
        ypred = search.predict_proba(X_test)
        result = roc_auc_score(y_test, yhat)
        acc = accuracy_score(y_test, yhat)
        print("{:.2f}".format(result)) 



        
        ## Use SHAP to explain predictions
        ### prepare preprocessed X_test 
        cloned_preproc_pipe = Pipeline([search.best_estimator_.steps[0], search.best_estimator_.steps[3]])
        X_test_transformed= pd.DataFrame(data= cloned_preproc_pipe.transform(X_test), columns=X.columns)
        
        
        explainer = shap.TreeExplainer(search.best_estimator_.steps[-1][1])
        shap_values = explainer.shap_values(X_test_transformed)

        # Extract SHAP information per fold per sample 
        for i, test_index in enumerate(test_outer_ix):
            shap_values_per_cv[test_index][CV_repeat] = shap_values[i] #-#-#


        # Append
        results.append({'count':count,'yhat': 
                        yhat,'ypred':ypred,
                        'ytest':y_test,
                        'score_acc':[acc]})
    
        y_pred = pd.DataFrame(ypred.argmax(axis=1))

        conf_matrix = classification_report(list(y_test),list(ypred.argmax(axis=1)),target_names=labels,output_dict=True)
        #conf_matrix = classification_report(list(y_test),(list(y_pred[0])),target_names = labels, output_dict=True)
        # save classification_report for obtaining precision,recall,f1 and accuracy for later 
        conf_matrix_list_of_arrays.append({'count':count,
                                          'macro-avg':conf_matrix['macro avg'],
                                           labels[1]:conf_matrix[labels[1]],
                                           labels[0]:conf_matrix[labels[0]],
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


os.chdir(directory)


plt.tight_layout()
plt.savefig("ROC.png",dpi=600)
#plt.show()
plt.close()


resultsdf = pd.DataFrame(results)


            
            
            
            
# Establish lists to keep average Shap values, their Stds, and their min and max
average_shap_values, stds, ranges = [],[],[]

for i in range(0,len(X)):
    df_per_obs = pd.DataFrame.from_dict(shap_values_per_cv[i]) # Get all SHAP values for sample number i (n(X) x N_REPEATS)
    # Get relevant statistics for every sample 
    average_shap_values.append(df_per_obs.mean(axis=1).values) 
    stds.append(df_per_obs.std(axis=1).values)
    ranges.append(df_per_obs.max(axis=1).values-df_per_obs.min(axis=1).values)           
            
            

            
            
shap_vals = np.array(average_shap_values)
avg_shap_values = pd.DataFrame(shap_vals, columns=X.columns)
avg_shap_values.to_csv("avg_shap_values.csv",encoding="utf-8", sep="\t")

stds_vals = np.array(stds)
stds_df = pd.DataFrame(stds_vals, columns=X.columns)
stds_df.to_csv("stds_shap.csv",encoding="utf-8", sep="\t")


ranges_vals = np.array(ranges)
ranges_df = pd.DataFrame(ranges_vals, columns=X.columns)
ranges_df.to_csv("ranges_shap.csv",encoding="utf-8", sep="\t")
           

            
end_time = datetime.now()
print('Duration: {}'.format(end_time - start_time))
        
            
            
            
            
            


  
#####----#####----#####----#####
#### OPTIONAL SHAP PLOTS    ####
#####----#####----#####----#####

l0 = labels[0].replace(" ","_")
l1 = labels[1].replace(" ","_")





sizes = [3, 5, 10, 15]
for n in sizes:

    ### bee
    fig, ax = plt.subplots(figsize=(10,10),dpi=600) 
    fig = shap.summary_plot(np.array(average_shap_values), X, max_display=n, show = False)
    ax.tick_params(axis='y', labelsize=14)
    ax.tick_params(axis='x', labelsize=14)
    plt.xlabel('SHAP value (impact on model output) ', fontsize=16)
    title ="SHAP summary of "+str(n)+" most important features"
    plt.title(title, fontsize=16)
    title = "SHAP_bee_{:.0f}_{}_{}.png".format(n,l0,l1)
    plt.savefig(title,bbox_inches="tight")
    #plt.show()
    plt.close()
    
    ### bar
    fig, ax = plt.subplots(figsize=(10,10),dpi=600) 
    fig = shap.summary_plot(np.array(average_shap_values), X, max_display=n, plot_type="bar", show = False)
    ax.tick_params(axis='y', labelsize=14)
    ax.tick_params(axis='x', labelsize=14)
    plt.xlabel('average impact on model output magnitude \n mean (|SHAP|) value ', fontsize=16)
    title ="SHAP summary of "+str(n)+" most important features"
    plt.title(title, fontsize=16)
    title = "SHAP_bar_{:.0f}_{}_{}.png".format(n,l0,l1)
    plt.savefig(title,bbox_inches="tight")
    #plt.show()
    plt.close()
    
     
            
            
            
            
            


from sklearn.metrics import confusion_matrix, plot_confusion_matrix, ConfusionMatrixDisplay
tn_sve= "cm {} vs {}.png".format(labels[0],labels[1])
cm = confusion_matrix(np.concatenate(resultsdf["ytest"]),np.concatenate(resultsdf["yhat"]))
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                              display_labels={labels[0],labels[1]})
fig = disp.plot()
plt.title("confusion matrix")
#plt.show()
plt.savefig(tn_sve,dpi=300)
plt.close()




from sklearn.metrics import average_precision_score, precision_recall_curve
from sklearn.metrics import auc, plot_precision_recall_curve

# get y_test 
yt = np.concatenate(resultsdf["ytest"])
# get y_pred of #class 1
yp = np.concatenate(resultsdf["ypred"])
yp = yp[:,1]

# get avg precision + precision & recall scores
average_precision = average_precision_score(yt,yp)
precision, recall, thresholds = precision_recall_curve(yt, yp)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.grid()

# plot recall and precision
fig = plt.plot(recall, precision)
# fill area under the curve 
plt.fill_between(recall, precision, alpha=0.20)

tn_sve= "binary class PR AUC {} vs {}.png".format(labels[0],labels[1])

# set title 
plt.title('binary class precision recall AUC: {0:0.2f}'.format(average_precision), fontsize = 13)

# set labels and font size
ax.set_xlabel('recall/True positive rate', fontsize = 12)
ax.set_ylabel('precision', fontsize = 12)
plt.savefig(tn_sve,dpi=400)
plt.close()









from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from matplotlib import pyplot
# generate a no skill prediction (majority class)
ns_probs = [1 for _ in range(len(np.concatenate(resultsdf["ytest"])))]

# keep probabilities for the positive outcome only
lr_probs = np.concatenate(resultsdf["ypred"])[:,1]
# calculate scores
ns_auc = roc_auc_score(np.concatenate(resultsdf["ytest"]), ns_probs)
lr_auc = roc_auc_score(np.concatenate(resultsdf["ytest"]), lr_probs)
# summarize scores
print('No Skill: ROC AUC=%.3f' % (ns_auc))
print('XGB: ROC AUC=%.3f' % (lr_auc))
# calculate roc curves
ns_fpr, ns_tpr, _ = roc_curve(np.concatenate(resultsdf["ytest"]), ns_probs)
lr_fpr, lr_tpr, _ = roc_curve(np.concatenate(resultsdf["ytest"]), lr_probs)
# plot the roc curve for the model
ax = plt.plot(ns_fpr, ns_tpr, linestyle='--', label='Random')
ax = plt.plot(lr_fpr, lr_tpr, marker='.', label=labels[0])
# axis labels
ax = plt.xlabel('False Positive Rate')
ax = plt.ylabel('True Positive Rate')
plt.title('binary class precision recall AUC: {0:0.2f}'.format(average_precision), fontsize = 13)

tn_sve= "TP FP {} vs {} matplot.png".format(labels[0],labels[1])

# show the legend
ax = plt.legend()
# show the plot
#pyplot.show()
plt.savefig(tn_sve,dpi=400)
plt.close()










# precision-recall curve and f1
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import f1_score
from sklearn.metrics import auc
from matplotlib import pyplot

# keep probabilities for the positive outcome only
lr_probs = np.concatenate(resultsdf["ypred"])[:,1]
# predict class values
yhat = np.concatenate(resultsdf["yhat"])
testy = np.concatenate(resultsdf["ytest"])

lr_precision, lr_recall, _ = precision_recall_curve(testy, lr_probs)
lr_f1, lr_auc = f1_score(testy, yhat), auc(lr_recall, lr_precision)
# summarize scores
print('Logistic: f1=%.3f auc=%.3f' % (lr_f1, lr_auc))
# plot the precision-recall curves
no_skill = len(testy[testy==1]) / len(testy)
ax = plt.plot([0, 1], [no_skill, no_skill], linestyle='--', label='Null')
ax = plt.plot(lr_recall, lr_precision, marker='.', label=labels[1], color="b")
# axis labels
ax = plt.xlabel('Recall')
ax = plt.ylabel('Precision')
ax = plt.title("Precision Recall Curve")
# show the legend
ax = plt.legend()
# show the plot
#pyplot.show()

tn_sve= "binary class PR AUC {} vs {} matplot.png".format(labels[0],labels[1])
plt.savefig(tn_sve,dpi=400)






# convert to dataframe
resultsdf = pd.DataFrame(results)

ypreds = list((np.concatenate(resultsdf["ypred"])).argmax(axis=1))
print(classification_report(list(np.concatenate(resultsdf["ytest"])),ypreds,target_names=labels))

report = classification_report(list(np.concatenate(resultsdf["ytest"])),ypreds,target_names=labels, output_dict=True)
df = pd.DataFrame(report).transpose()
df.to_csv("classification_report.csv", sep="\t", encoding="utf-8")







# save results
resultsdf = pd.DataFrame(results)
resultsdf.to_csv("results.csv",encoding="utf-8",sep="\t")

tprs_df = pd.DataFrame(data=tprs)
tprs_df.to_csv("tprs.csv", encoding="utf-8",sep="\t")

# save model 
import pickle
import joblib

bm= search.best_estimator_.steps[-1][1]
pickle.dump(bm, open("xgb_model.pkl", "wb"))


# save

filename = 'grid_searched_model.sav'

model = search.best_estimator_.steps[-1][1]
joblib.dump(model, filename)


with open("conf_matrix_list_of_arrays.pkl","wb") as fp:
  pickle.dump(conf_matrix_list_of_arrays,fp)


import pickle
with open("avgshap_values.pkl","wb") as fp:
     pickle.dump(shap_vals,fp)






import sys 
sys.exit()
  

