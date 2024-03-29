# -*- coding: utf-8 -*-


import os
import pandas as pd
from sklearn.impute import SimpleImputer, MinMaxScaler, StandardScaler
import statsmodels.api as sm

###############################################################################################
# modified (see below) with age & gender as constant variables which are kept each iteration
###############################################################################################



#import statsmodels as sm
import statsmodels.api as sm


def stepwise_selection(X, y, 
                       initial_list=[], 
                       threshold_in=0.05, 
                       threshold_out = 0.06, 
                       verbose=True):
    """ Perform a forward-backward feature selection 
    based on p-value from statsmodels.api.OLS
    planspace.org/20150423-forward_selection_with_statsmodels/
    Arguments:
        X - pandas.DataFrame with candidate features
        y - list-like with the target
        initial_list - list of features to start with (column names of X)
        threshold_in - include a feature if its p-value < threshold_in
        threshold_out - exclude a feature if its p-value > threshold_out
        verbose - whether to print the sequence of inclusions and exclusions
    Returns: list of selected features 
    Always set threshold_in < threshold_out to avoid infinite looping.
    See https://en.wikipedia.org/wiki/Stepwise_regression for the details
    """
   
    initial_list = []
    best_feats = []
    worst_feats = []
    
    included = list(initial_list)
    X_ =X.drop(["group","age","gender"],axis=1)

    while True:
       changed=False
      
       # forward step
       excluded = list(set(X_.columns)-set(included))
       new_pval = pd.Series(index=excluded)
       for new_column in excluded:
           # model = sm.OLS(y, sm.add_constant(pd.DataFrame(X[included+[new_column]]))).fit()
           formul = 'group ~ {} + age + gender'.format(' + '.join(included + [new_column]))
           model = sm.formula.logit(formula=str(formul),data=X).fit()
           print(formul)
           new_pval[new_column] = model.pvalues[new_column]
           print("{}".format(new_pval[new_column]))
       best_pval = new_pval.min()
       print("best pval: {}".format(np.round(best_pval,4)))
       if best_pval < threshold_in:
           best_feature = new_pval.argmin()
           included.append(new_pval.index[best_feature])
           # best_feats[new_pval.index[best_feature]] = ((new_pval[best_feature],new_pval.index[best_feature]))
           best_feats.append((new_pval[best_feature],new_pval.index[best_feature]))

           changed=True
           if verbose:
               print('Add {} with p-value {} '.format(best_feature, best_pval))

       if not changed:
           print("break")
           break

       # backward step
       ## model = sm.OLS(y, sm.add_constant(pd.DataFrame(X[included]))).fit()
       formul = 'group ~ {} + age + gender'.format(' + '.join(X[included]))
       model = sm.formula.logit(formula=str(formul),data=X).fit()
       # use all coefs except intercept
       pvalues = model.pvalues.iloc[1:]
       worst_pval =(( pvalues.index[pvalues.argmax()], pvalues.max() )) # null if pvalues is empty
       # worst_feats = model.pvalues[pvalues.argmax()]
       
       if (worst_pval[1] > threshold_out) & ((worst_pval[0] != 'gender') & (worst_pval[0] !=  'age') ):
          changed=True
          #worst_feature = pvalues.argmax()
          worst_feature = pvalues.index[pvalues.argmax()]
          worst_feats.append(( pvalues.index[pvalues.argmax()], pvalues[worst_feature] ))
          included.remove(worst_feature)
          if verbose:
             print('Drop {} with p-value {}'.format(worst_feature, worst_pval))
       if not changed:
            break
    return included
