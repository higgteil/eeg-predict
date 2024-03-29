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


#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
# performs repeated nested cross validation by domain (eeg, neuropsych, clinical, socio-environmental)    #
# losely based on towardsdatascience.com/using-shap-with-cross-validation-d24af548fadc                    #
#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#


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
import shap
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
from sklearn.impute import KNNImputer
from imblearn.under_sampling import RandomUnderSampler
import glob



def load_X():
    direc= "./"
    os.chdir(direc)
    X = pd.read_csv(glob.glob("*.csv")[0],encoding="utf-8", sep="\t", index_col=0).reset_index(drop=True)
    y = X.groups_hs_ls
    ## "positive" class
    y = y.replace({0:1,1:0})
    X = X.drop(X.filter(regex="group|ID").columns.to_list(),axis=1)
    return X,y


X,y = load_X()

eeg = X.filter(regex=r"q_[A-Z]z(?=[\w+])|Average").columns.to_list()
neuropsych = X.filter(regex="cpt|fwit|vlm|bls|bzt|vfl|mwt|zsn").columns.to_list()
clinical = X.filter(regex=r"audit|adhd|psqi|PSQI|STAI|stai|tpq|fev|neo|rrs|rrd|haendig|bdi_|pss|BMI|psychiatrisch|soz_10|soz_11|soz_12|qsu|BDI|PSS|mean|pack|ftnd|onset").columns.to_list()
socio_environmental = X.filter(regex=r"vater|mutter|geschw|umfeld|familie|beruf|schule|soz_03|soz_04|soz_05|soz_06|soz_07|soz_08|soz_09|gender|^age").columns.to_list()

all_categories = [eeg, neuropsych, clinical, socio_environmental]
all_categories_dict= ({"EEG":eeg, "neuropsychological":neuropsych, "clinical": clinical, "sociodemographic-environmental": socio_environmental})


import warnings 
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use('Agg')


from imblearn import FunctionSampler
from imblearn.pipeline import make_pipeline

def outlier_rejection(X, y):
    """This will be our function used to resample our dataset."""
    model = IsolationForest(max_samples=1, contamination="auto", random_state=1510)
    model.fit(X)
    y_pred = model.predict(X)
    return X[y_pred == 1], y[y_pred == 1]

reject_sampler = FunctionSampler(func=outlier_rejection)



import sys
import builtins
import os

fn = "fnlog.txt"
sys.stdout = open(fn, "w", buffering=1)
def print(text):
    builtins.print(text)
    os.fsync(sys.stdout)
    
    
    

### iterate over categories 

for key, value in all_categories_dict.items():

    print(key)
    X,y = load_X()
    X = X[value]
    

    # Reproducibility
    np.random.seed(1)  

    # repetitions 
    N_CV_REPEATS = 5


    from datetime import datetime
    start_time = datetime.now()

    import time
    timestr = time.strftime("%Y-%m-%d-%H-%M-%S")
    fname = "nestedCV_"+str(N_CV_REPEATS)+"_repeats_"+timestr+"___"+key
    directory = "./per_domain/{}_{}".format(key,timestr)
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



    SEED = 1510
    count = 0

    labels=['Heavy Smokers','Light Smokers']

    import matplotlib.pyplot as plt
    import numpy as np
    # save // plot stuff
    fig, ax = plt.subplots()
    outer_results, conf_matrix_list_of_arrays, tprs, aucs, results = [],[],[],[],[]
    base_fpr = np.linspace(0, 1, 101)
    mean_fpr = np.linspace(0, 1, 100)



    for i, CV_repeat in enumerate(range(N_CV_REPEATS)): 

        #Establish CV scheme
        CV = StratifiedKFold(n_splits=10, shuffle=True, random_state=random_states[i]) 
        print("fold {} and repeat {}".format(i,CV_repeat))
        

        ix_training, ix_test = [], []
        # Loop through each fold and append the training & test indices to the empty lists above
        for fold in CV.split(X,y):
            ix_training.append(fold[0]), ix_test.append(fold[1])
            
        ## Loop through each outer fold and extract SHAP values 
        for i, (train_outer_ix, test_outer_ix) in enumerate(zip(ix_training, ix_test)): 
            #Verbose
            print('Fold Number:{}'.format(i))
            X_train, X_test = X.iloc[train_outer_ix, :], X.iloc[test_outer_ix, :]
            y_train, y_test = y.iloc[train_outer_ix], y.iloc[test_outer_ix]

           
            ## Establish inner CV for parameter optimization
            cv_inner = StratifiedKFold(n_splits=5, shuffle=True, random_state=1510)

            pipeline = imbpipeline(steps = [
                                        ['imputer',SimpleImputer(strategy="most_frequent")],
                                        ['smote',SMOTETomek()],
                                        ['outlier',reject_sampler],
                                        ['scaler', MinMaxScaler()],
                                        ['classifier', XGBClassifier(verbose=0)]
                                        ])

                    
            params = {
                      'classifier__max_depth':[4, 6, 8, 10],
                      'classifier__learning_rate': [.01,.1,.2,.3],
                      'classifier__subsample': [.7, .8, .9,1],
                      'classifier__gamma':[0,.1,.3,.5,1],
                      'classifier__colsample_bytree': [.6, .8, 1.0]
                    }
                
                    
            # Search to optimize hyperparameters
            search = RandomizedSearchCV(pipeline, params, cv=cv_inner) 
            search.fit(X_train, y_train)

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
                shap_values_per_cv[test_index][CV_repeat] = shap_values[i] 


            # Append
            results.append({'count':count,'yhat': 
                            yhat,'ypred':ypred,
                            'ytest':y_test,
                            'score_acc':[acc]})
        
            y_pred = pd.DataFrame(ypred.argmax(axis=1))

            conf_matrix = classification_report(list(y_test),list(ypred.argmax(axis=1)),target_names=labels,output_dict=True)
            # save classification_report for obtaining precision,recall,f1 and accuracy for later 
            conf_matrix_list_of_arrays.append({'count':count,
                                              'macro-avg':conf_matrix['macro avg'],
                                              labels[0]:conf_matrix[labels[0]],
                                              labels[1]:conf_matrix[labels[1]],
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
            
            ## print log
            print("randomized gs params:{}".format(search.best_estimator_[-1].get_params()))
            print("CV repeat: {:.0f}, fold: {:.0f}, train:{:.3f}, test: {:.3f}".format(CV_repeat, i, search.score(X_train,y_train),result))


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

    tn = "ROC classification {} vs {} \n domain {}".format(labels[0],labels[1],key)


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
    fn = "avg_shap_values_{}.csv".format(key)
    avg_shap_values.to_csv(fn,encoding="utf-8", sep="\t")

    stds_vals = np.array(stds)
    stds_df = pd.DataFrame(stds_vals, columns=X.columns)
    fn = "stds_shap_values_{}.csv".format(key)
    stds_df.to_csv(fn,encoding="utf-8", sep="\t")


    ranges_vals = np.array(ranges)
    ranges_df = pd.DataFrame(ranges_vals, columns=X.columns)
    fn = "ranges_shap_values_{}.csv".format(key)
    ranges_df.to_csv(fn,encoding="utf-8", sep="\t")
              

                
    end_time = datetime.now()
    print('Duration: {}'.format(end_time - start_time))
            
    
    # convert to dataframe
    resultsdf = pd.DataFrame(results)

    ypreds = list((np.concatenate(resultsdf["ypred"])).argmax(axis=1))
    print(classification_report(list(np.concatenate(resultsdf["ytest"])),ypreds,target_names=labels))

    report = classification_report(list(np.concatenate(resultsdf["ytest"])),ypreds,target_names=labels, output_dict=True)
    df = pd.DataFrame(report).transpose()
    fn = "results_{}.csv".format(key)
    df.to_csv(fn, sep="\t", encoding="utf-8")


    # save results
    resultsdf = pd.DataFrame(results)
    fn = "results_{}.csv".format(key)
    resultsdf.to_csv(fn,encoding="utf-8",sep="\t")

    tprs_df = pd.DataFrame(data=tprs)
    tprs_df.to_csv("tprs.csv", encoding="utf-8",sep="\t")

    # save model 
    import pickle
    import joblib
    bm= search.best_estimator_.steps[-1][1]
    pickle.dump(bm, open("xgb_model.pkl", "wb"))

    filename = 'grid_searched_model.sav'

    model = search.best_estimator_.steps[-1][1]
    joblib.dump(model, filename)


    with open("conf_matrix_list_of_arrays.pkl","wb") as fp:
      pickle.dump(conf_matrix_list_of_arrays,fp)

    import pickle
    fn = "avgshap_values_{}.pkl".format(key)
    with open(fn,"wb") as fp:
        pickle.dump(shap_vals,fp)
