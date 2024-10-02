#!/usr/bin/env python
# coding: utf-8

# In[1]:


from google.colab import drive
drive.mount('/content/drive/')


# In[2]:


get_ipython().system('pip install shap xgboost imblearn')


# In[3]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.ensemble import RandomForestClassifier
from sklearn.base import clone
from typing import List
import matplotlib.colors as mcolors

import numpy as np
import pandas as pd

from imblearn import FunctionSampler
from imblearn.pipeline import make_pipeline

from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.impute import SimpleImputer
from sklearn.metrics import *
from imblearn.pipeline import Pipeline as imbpipeline
from imblearn.combine import SMOTETomek
from xgboost import XGBClassifier
from sklearn.ensemble import IsolationForest
from imblearn.pipeline import make_pipeline
from imblearn.base import FunctionSampler
import shap
import warnings
import os
import time
import shutil
from tensorflow.keras.models import load_model


# In[4]:


def plot_roc_curve(mean_fpr, tprs, aucs, labels):
    """
    Plot the ROC curve.

    Parameters:
    mean_fpr (array): Mean false positive rate.
    tprs (list): List of true positive rates.
    aucs (list): List of AUC scores.
    labels (list): Class labels.

    Returns:
    Figure: The ROC curve plot.
    """
    from sklearn.metrics import auc

    mean_tpr = np.mean(tprs, axis=0)
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)

    plt.figure()
    plt.plot(mean_fpr, mean_tpr, color='b',
             label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
             lw=2, alpha=.8)

    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', alpha=.8)

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.savefig('ROC.png', dpi=600)
    return plt.gcf()



# In[5]:


def create_modified_shap_plot(model, X, max_display=10, **kwargs):
    try:
        explainer = shap.Explainer(model)
        shap_values = explainer(X)

        # Compute the absolute SHAP values
        abs_shap_values = np.abs(shap_values.values)

        # Plot the regular "dot" summary plot and get the y-tick positions
        shap.summary_plot(shap_values, X, plot_type="dot", show=False, max_display=max_display)
        y_positions = [tick.get_position()[1] for tick in plt.gca().get_yticklabels()]
        plt.close()

        # Plot the beeswarm using absolute SHAP values
        shap.plots.beeswarm(shap.Explanation(abs_shap_values, shap_values.base_values,
                                shap_values.data, feature_names=shap_values.feature_names),
                                show=False, max_display=max_display, **kwargs)

        # Get the current ax to modify
        ax = plt.gca()

        # Calculate median values for the displayed features
        feature_order = np.argsort(np.sum(abs_shap_values, axis=0))[::-1]
        displayed_medians = [np.median(abs_shap_values[:, idx]) for idx in feature_order[:max_display-1]]

        # Adjusting the remaining_values computation
        combined_remaining = np.sum(abs_shap_values[:, feature_order[max_display-1:]], axis=1)
        displayed_medians.append(np.median(combined_remaining))

        # Scatter median values on top of the beeswarm plot, using the y_positions from the "dot" summary plot
        ax.scatter(displayed_medians, y_positions[::-1], color='k', marker='|', s=150, zorder=10)

        if os.path.exists(directory):
           os.chdir(directory)
        else:
          os.makedirs(directory, exist_ok=True)
        print("plot saved in:{}".format(directory))

        plt.gcf()
        plt.tight_layout()
        plt.savefig('modified_shap_plot.png')
        plt.close()

    except Exception as e:
           logger = SimpleCVLogger()
           logger.error(f"Error in plotting custom SHAP: {str(e)}")

    return
    pass


def compute_aggregated_shap_nested_cv(X: pd.DataFrame, y: np.ndarray, search):
    """
    Compute aggregated SHAP values using nested cross-validation.

    :param X: Feature matrix
    :param y: Target vector
    :param model: Base model to use (will be cloned for each fold)
    :param n_outer_splits: Number of splits for outer CV
    :param n_inner_splits: Number of splits for inner CV
    :param n_repeats: Number of times to repeat the entire nested CV process
    :return: Tuple of (aggregated SHAP values, feature names)
    """
    all_shap_values = []
    all_feature_names = X.columns.tolist()

    tuned_model = search.best_estimator_.steps[-1][1]
    explainer = shap.TreeExplainer(tuned_model)

    shap_values = explainer(X)
    if len(shap_values.shape) == 3:
        shap_values_abs = np.abs(shap_values.values).sum(axis=2)
    else:
        shap_values_abs = np.abs(shap_values.values)

    all_shap_values.append(shap_values_abs)

    # Aggregate SHAP values
    aggregated_shap = np.zeros((len(X), len(all_feature_names)))
    for shap_values in all_shap_values:
        aggregated_shap[:len(shap_values)] += shap_values
    aggregated_shap /= len(all_shap_values)



    return aggregated_shap, all_feature_names


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from typing import List

def create_modified_shap_plot_aggregated(aggregated_shap: np.ndarray,
                                         feature_names: List[str], directory: str,
                                         max_display: int = 10):
    """
    Create a modified SHAP beeswarm plot with aggregated values.
    :param aggregated_shap: Aggregated SHAP values
    :param feature_names: List of feature names
    :param max_display: Maximum number of features to display
    :return: Matplotlib figure
    """

    try:
        # Ensure max_display is an integer
        assert isinstance(max_display, int), "max_display should be an integer"


        feature_importance = np.mean(np.abs(aggregated_shap), axis=0)
        sorted_idx = np.argsort(feature_importance)
        sorted_features = [feature_names[i] for i in sorted_idx[::-1]]

        plot_data = pd.DataFrame(aggregated_shap, columns=feature_names)
        plot_data = plot_data[sorted_features[:max_display]]

        fig, ax = plt.subplots(figsize=(10, 8))

        # Create a custom colormap
        base_color = "#E0115F"  # Ruby red
        light_color = mcolors.to_rgba(base_color, alpha=0.7)
        dark_color = mcolors.to_rgba(base_color, alpha=1.0)
        custom_cmap = mcolors.LinearSegmentedColormap.from_list("custom", [light_color, dark_color])

        for i, feature in enumerate(plot_data.columns):
            values = plot_data[feature]
            y = np.full_like(values, i)
            y_jitter = y + np.random.normal(0, 0.1, size=len(y))

            # Normalize SHAP values for coloring
            normalized_values = (np.abs(values) - np.min(np.abs(values))) / (np.max(np.abs(values)) - np.min(np.abs(values)))

            # Use the custom colormap
            colors = custom_cmap(normalized_values)

            # Plot dots without edges, using the original style
            ax.scatter(values, y_jitter, c=colors, s=20, edgecolors='none')

        medians = plot_data.median().values
        ax.scatter(medians, range(len(medians)), color='black', marker='|', s=150, zorder=10)

        ax.set_yticks(range(len(plot_data.columns)))
        ax.set_yticklabels(plot_data.columns)

        ax.set_xlabel('SHAP value (impact on model output)')
        ax.set_title("Aggregated SHAP Values Across Nested CV")

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        if os.path.exists(directory):
           os.chdir(directory)
        else:
          os.makedirs(directory, exist_ok=True)
        print("plot saved in:{}".format(directory))

        plt.tight_layout()
        plt.savefig('aggregated_shap_plot.png')
        plt.close(fig)

        return fig
    except Exception as e:
        print(f"Error in plotting custom SHAP: {str(e)}")
        return None


# In[6]:


import numpy as np
import pandas as pd

def generate_mock_data(n_samples=1000):
    np.random.seed(42)

    # EEG bands and electrodes
    bands = ['Alpha', 'Beta1', 'Beta2', 'Gamma', 'Delta', 'Theta']
    electrodes = ['Fp1', 'Fp2','A1','A2','F3','F4', 'F7', 'F8', 'C4', 'C3','T5', 'T6','P3','P4', 'O1', 'O2', 'Cz', 'Pz', 'Fz']

    # Generate feature names for EEG data
    feature_names = [f'{electrode}_{band}' for band in bands for electrode in electrodes]

    # Additional variables
    additional_vars = ['q_Fz_P50', 'q_Cz_P50', 'q_Pz_P50', 'q_Fz_gamma', 'q_Cz_gamma', 'q_Pz_gamma',
                       'q_Fz_theta', 'q_Cz_theta', 'q_Pz_theta', 'q_Fz_alpha', 'q_Cz_alpha', 'q_Pz_alpha']

    feature_names.extend(additional_vars)

    # Calculate total number of features
    n_features = len(feature_names)

    # Generate mock data
    X = pd.DataFrame(np.random.randn(n_samples, n_features), columns=feature_names)

    # Generate target variable
    y = pd.Series(np.random.randint(0, 2, n_samples), name='target')

    # Convert column names to strings
    X.columns = X.columns.astype(str)

    return X, y

# Example usage
#X, y = generate_mock_data(70)
#print(f"Shape of X: {X.shape}")
#print(f"Shape of y: {y.shape}")
#print("\nFirst few columns of X:")
#print(X.iloc[:, :5].head())
#print("\nFirst few values of y:")
#print(y.head())


# In[7]:


import json
from datetime import datetime
import os
import numpy as np
import matplotlib.pyplot as plt
import shap
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, classification_report

class SimpleCVLogger:
    def __init__(self, results_file="cv_results.json", conf_matrix_dir="conf_matrices", directory=None):
        self.results_file = results_file
        self.conf_matrix_dir = conf_matrix_dir
        self.results = []
        self.directory = directory
        if directory:
            os.makedirs(self.directory, exist_ok=True)
        os.makedirs(self.conf_matrix_dir, exist_ok=True)

    def log_result(self, repeat, fold, y_true, y_pred, y_pred_proba, tuned_model, labels):
        best_params_dict = tuned_model.best_params_
        conf_matrix = classification_report(y_true, y_pred, target_names=labels, output_dict=True)

        result = {
            'timestamp': datetime.now().isoformat(),
            'repeat': repeat,
            'fold': fold,
            'accuracy': accuracy_score(y_true, y_pred),
            'balanced_accuracy': balanced_accuracy_score(y_true, y_pred),
            'f1_score': f1_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred),
            'recall': recall_score(y_true, y_pred),
            'roc_auc': roc_auc_score(y_true, y_pred_proba[:, 1]),
            'best_params': best_params_dict,
        }

        # Round all float values to 2 decimal places
        for key, value in result.items():
            if isinstance(value, float):
                result[key] = round(value, 2)

        self.results.append(result)
        self._save_results()
        self._save_conf_matrix(repeat, fold, conf_matrix)

        return result

    def _save_results(self):
        with open(self.results_file, 'w') as f:
            json.dump(self.results, f, indent=2)

    def _save_conf_matrix(self, repeat, fold, conf_matrix):
        filename = f"conf_matrix_repeat{repeat}_fold{fold}.json"
        filepath = os.path.join(self.conf_matrix_dir, filename)
        with open(filepath, 'w') as f:
            json.dump(conf_matrix, f, indent=2)

    def get_results(self):
        return self.results

    def error(self, message):
        print(f"Error: {message}")



def create_modified_shap_plot(model, X, directory: str, max_display: int = 10, **kwargs):
    try:
        max_display = int(max_display)
        explainer = shap.Explainer(model)
        shap_values = explainer(X)

        # Compute the absolute SHAP values
        abs_shap_values = np.abs(shap_values.values)

        # Plot the regular "dot" summary plot and get the y-tick positions
        shap.summary_plot(shap_values, X, plot_type="dot", show=False, max_display=max_display)
        y_positions = [tick.get_position()[1] for tick in plt.gca().get_yticklabels()]
        plt.close()

        # Plot the beeswarm using absolute SHAP values
        shap.plots.beeswarm(shap.Explanation(abs_shap_values, shap_values.base_values,
                                            shap_values.data, feature_names=shap_values.feature_names),
                            show=False, max_display=max_display, **kwargs)

        # Get the current ax to modify
        ax = plt.gca()

        # Calculate median values for the displayed features
        feature_order = np.argsort(np.sum(abs_shap_values, axis=0))[::-1]
        displayed_medians = [np.median(abs_shap_values[:, idx]) for idx in feature_order[:max_display-1]]

        # Adjusting the remaining_values computation
        combined_remaining = np.sum(abs_shap_values[:, feature_order[max_display-1:]], axis=1)
        displayed_medians.append(np.median(combined_remaining))

        # Scatter median values on top of the beeswarm plot, using the y_positions from the "dot" summary plot
        ax.scatter(displayed_medians, y_positions[::-1], color='k', marker='|', s=150, zorder=10)

        if os.path.exists(directory):
          os.chdir(directory)
        else:
          os.makedirs(directory, exist_ok=True)
        print("plot saved in:{}".format(directory))

        plt.tight_layout()
        plt.savefig('modified_shap_plot.png')
        plt.close()

    except Exception as e:
        logger = SimpleCVLogger()
        logger.error(f"Error in plotting custom SHAP: {str(e)}")




def data_dump(data_to_save, directory):
    #import os, json
    import os
    import json
    # Create the directory if it doesn't exist
    if os.path.exists(directory):
       os.chdir(directory)
    else:
       os.makedirs("./dumpdata", exist_ok=True)
  # dump data
  # Iterate through the data and save each to a separate file
    for filename, data in data_to_save.items():
        with open(os.path.join(directory, f"{filename}.json"), 'w') as f:
            json.dump(data, f)
    pass


# In[ ]:





# In[ ]:


os.chdir("/content/drive/MyDrive/results/Quitters_vs_Nonquitters/updated/test11/ncv_1_repeats_QNQ_test2024-10-01-14-19-04")



# In[ ]:


import json

# If the file is uploaded to Colab
with open('results.json', 'r') as file:
    data = json.load(file)

# If you're reading from a URL
# import requests
# url = "https://example.com/your_file.json"
# response = requests.get(url)
# data = json.loads(response.text)

# Now 'data' contains the contents of your JSON file
print(data)


# In[8]:


import numpy as np
import pandas as pd
import shap

def aggregate_shap_values(model, X_test, test_outer_ix, idx_repeat, idx_fold):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    if isinstance(shap_values, list):  # For multi-class problems
        shap_values = np.abs(np.array(shap_values)).mean(axis=0)
    else:
        shap_values = np.abs(shap_values)

    return pd.DataFrame({
        'index': test_outer_ix,
        'repeat': idx_repeat,
        'fold': idx_fold,
        'shap_values': list(shap_values)
    })


# In[9]:


def outlier_rejection(X, y, max_samples='auto', contamination='auto', random_state=1510):
    try:
        if not isinstance(X, np.ndarray) and not hasattr(X, 'iloc'):
            raise ValueError("X must be a numpy array or a pandas DataFrame")
        if not isinstance(y, np.ndarray) and not hasattr(y, 'iloc'):
            raise ValueError("y must be a numpy array or a pandas Series")
        if len(X) != len(y):
            raise ValueError("X and y must have the same number of samples")

        model = IsolationForest(max_samples=max_samples, contamination=contamination, random_state=random_state)
        model.fit(X)
        y_pred = model.predict(X)
        mask = y_pred == 1
        return X[mask], y[mask]
    except Exception as e:
        logger.error(f"Error in outlier rejection: {str(e)}")
        return X, y

reject_sampler = FunctionSampler(func=outlier_rejection)


# In[10]:


warnings.filterwarnings("ignore")

class Autoencoder:
    def __init__(self):
        self.scaler = MinMaxScaler()
        self.imputer = SimpleImputer(strategy="most_frequent")
        self.encoder = self.load_encoder()

    @staticmethod
    def load_encoder():
        os.chdir("/content/drive/MyDrive/try/x19/")
        return load_model('encoder_model23.h5')

    def encode(self, X_train, X_test):
        eeg_cols = [col for col in X_train.columns if any(col.endswith(s) for s in ["Alpha", "Beta1", "Beta2", "Gamma", "Delta", "Theta"])
                    or (col.startswith("q_") and not col.startswith(("A2", "A1", "Coh_Frontal", "Coh_mean", "PSD_mean")))]

        X_train_eeg, X_test_eeg = X_train[eeg_cols], X_test[eeg_cols]

        X_train_eeg = pd.DataFrame(self.imputer.fit_transform(self.scaler.fit_transform(X_train_eeg)), columns=eeg_cols)
        X_test_eeg = pd.DataFrame(self.imputer.transform(self.scaler.transform(X_test_eeg)), columns=eeg_cols)

        X_train_f = pd.DataFrame(self.encoder.predict(X_train_eeg))
        X_test_f = pd.DataFrame(self.encoder.predict(X_test_eeg))
        X_train_encoded =  X_train.join(X_train_f)
        X_test_encoded =  X_test.join(X_test_f)
        X_train_encoded.columns = X_train_encoded.columns.astype(str)
        X_test_encoded.columns = X_test_encoded.columns.astype(str)

        return X_train_encoded, X_test_encoded

ae = Autoencoder()



def setup_directory(base_dir, n_cv_repeats):
    directory = os.path.join(base_dir, f"ncv_{n_cv_repeats}_repeats_QNQ_test{time.strftime('%Y-%m-%d-%H-%M-%S')}")
    os.makedirs(directory, exist_ok=True)
    return directory

def save_notebook(source_dir, nbn, destination_dir):
    shutil.copy(os.path.join(source_dir, nbn), os.path.join(destination_dir, f"used_code_{time.strftime('%Y-%m-%d-%H-%M-%S')}.ipynb"))

def outlier_rejection(X, y):
    model = IsolationForest(max_samples=1, contamination="auto", random_state=1510)
    y_pred = model.fit_predict(X)
    return X[y_pred == 1], y[y_pred == 1]

reject_sampler = FunctionSampler(func=outlier_rejection)



def create_pipeline():
    return imbpipeline([
        ('imputer', SimpleImputer(strategy="most_frequent")),
        ('smote', SMOTETomek(sampling_strategy="auto")),
        ('outlier', FunctionSampler(func=outlier_rejection)),
        ('classifier', XGBClassifier())
    ])

def get_hyperparameters():
    return {
        'classifier__max_depth': [4, 6, 8, 10],
        'classifier__learning_rate': [.01, .1, .2, .3],
        'classifier__subsample': [.7, .8, .9, 1],
        'classifier__gamma': [0, .1, .3, .5, 1],
        'classifier__colsample_bytree': [.6, .8, 1.0]
    }

def evaluate_model(y_true, y_pred, y_pred_proba):
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba[:, 1])
    return {'conf_matrix':classification_report(y_true,y_pred,target_names=["Quitters","Non-Quitters"], output_dict=True),
        'accuracy': accuracy_score(y_true, y_pred),
        'balanced_accuracy': balanced_accuracy_score(y_true, y_pred),
        'f1_score': f1_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred),
        'cohen_kappa': cohen_kappa_score(y_true, y_pred),
        'matthews_corr': matthews_corrcoef(y_true, y_pred),
        'auc': roc_auc_score(y_true, y_pred_proba[:, 1]),
        'fpr': fpr, 'tpr': tpr,
        'roc_auc': roc_auc_score(y_true, y_pred_proba[:, 1]),
    }

def plot_roc_curve(mean_fpr, tprs, aucs, labels, directory):
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1], linestyle="--", lw=2, color="r", label="Chance", alpha=0.8)
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    ax.plot(mean_fpr, mean_tpr, color="b", label=f"Mean ROC (AUC = {mean_auc:.2f} ± {std_auc:.2f})", lw=2, alpha=0.8)

    std_tpr = np.std(tprs, axis=0)
    ax.fill_between(mean_fpr, np.maximum(mean_tpr - std_tpr, 0), np.minimum(mean_tpr + std_tpr, 1), color="lightcoral", alpha=0.2, label=r"$\pm$ 1 std. dev.")
    ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05], title=f"ROC classification {labels[0]} vs {labels[1]}")
    ax.legend(loc="lower right")
    plt.tight_layout()
    if os.path.exists(directory):
        os.chdir(directory)
    else: pass
    plt.savefig('roc_curve.png')
    plt.close()
    return
    pass


# In[11]:


def nested_cross_validation(X, y, n_cv_repeats, random_states, ae, labels, directory):

    results, aucs, tprs, all_shap_values  = [], [], [], []
    shap_values_per_cv = {sample: {CV_repeat: {} for CV_repeat in range(n_cv_repeats)} for sample in X.index}
    mean_fpr = np.linspace(0, 1, 100)


    # INITIALIZE SHAP DICT FOR X_ENCODED
    ae=Autoencoder()
    X_enc, _ = ae.encode(X,X)
    all_feature_names = X_enc.columns.tolist()

    # Usage in your nested cross-validation loop:
    logger = SimpleCVLogger()


    # In the main nested cross-validation loop:
    for idx_repeat, random_state in enumerate(random_states):

        cv_outer = StratifiedKFold(n_splits=3, shuffle=True, random_state=random_state)
        cv_inner = StratifiedKFold(n_splits=2, shuffle=True, random_state=random_state)

        for idx_fold, (train_outer_ix, test_outer_ix) in enumerate(cv_outer.split(X, y)):
            print(f'\n------ Fold Number {idx_fold}')

            X_train, X_test = X.iloc[train_outer_ix, :], X.iloc[test_outer_ix, :]
            y_train, y_test = y.iloc[train_outer_ix], y.iloc[test_outer_ix]

            X_train_enc, X_test_enc = ae.encode(X_train, X_test)

            print("X_train_enc shape:{}, X_test_enc shape:{}".format(X_train_enc.shape, X_test_enc.shape))

            search = RandomizedSearchCV(create_pipeline(), get_hyperparameters(), cv=cv_inner, random_state=random_state)
            search.fit(X_train_enc, y_train)

            y_pred = search.predict(X_test_enc)
            y_pred_proba = search.predict_proba(X_test_enc)
            performance = evaluate_model(y_test, y_pred, y_pred_proba)
            performance.update({
                    'fold': idx_fold, 'repeat': idx_repeat,
                    'y_pred': y_pred_proba.argmax(axis=1), 'y_test': y_test, 'y_train': y_train,
                    'conf_matrix_arrays':performance['conf_matrix']
                })

            results.append(performance)
            aucs.append(performance['roc_auc'])
            tprs.append(np.interp(mean_fpr, performance['fpr'], performance['tpr']))
            print(f"Balanced Accuracy: {performance['balanced_accuracy']:.4f}")
            print(f"repeat cv {idx_repeat:.4f} fold {idx_fold:.0f}")

            model = search.best_estimator_.steps[-1][1]  # Get the final estimator
            fold_shap_values = aggregate_shap_values(model, X_test_enc, test_outer_ix, idx_repeat, idx_fold)
            all_shap_values.append(fold_shap_values)


           # for i, test_index in enumerate(test_outer_ix):
           #     shap_values_per_cv[test_index][idx_repeat] = all_shap_values[i]

    # Combine all SHAP values
    combined_shap_values = pd.concat(all_shap_values, ignore_index=True)
    # Calculate average SHAP values across all folds and repeats
    average_shap_values = combined_shap_values.groupby('index')['shap_values'].apply(lambda x: np.mean(np.vstack(x), axis=0))
    # If you need feature names
    feature_names = X_test_enc.columns  # Assuming X_test_enc is a DataFrame
    average_shap_df = pd.DataFrame(average_shap_values.tolist(), index=average_shap_values.index, columns=feature_names)

    # Reorder without transformation
    reordered_shap_values = combined_shap_values.groupby('index')['shap_values'].first().reset_index()

    # If you need to unnest the 'shap_values' column
    reordered_shap_df = pd.DataFrame(reordered_shap_values['shap_values'].tolist(),
                                    index=reordered_shap_values['index'],
                                    columns=feature_names)

    # If you need to preserve other columns like 'repeat' and 'fold'
    reordered_shap_values_with_meta = combined_shap_values.groupby('index').first().reset_index()


    result = logger.log_result(
             repeat=idx_repeat,
             fold=idx_fold,
             y_true=y_test,
             y_pred=y_pred,
             y_pred_proba=y_pred_proba,
             tuned_model=search,
             labels=labels
    )

    # Results are automatically saved after each fold, but you can access them if needed
    final_results = logger.get_results()
    print(f"Repeat {idx_repeat}, Fold {idx_fold}: ROC AUC = {result['roc_auc']:.4f}")


    fig1 = create_modified_shap_plot(model, X_enc, directory)
    fig2 = create_modified_shap_plot_aggregated(reordered_shap_df, feature_names, directory, max_display=10)
    fig3 = plot_roc_curve(mean_fpr, tprs, aucs, labels, directory)

    return final_results, result




def main():
    import os
    np.random.seed(1510)
    n_repeats = 1
    random_states = np.random.randint(10000, size=n_repeats)

    base_dir = '/content/drive/MyDrive/results/Quitters_vs_Nonquitters/updated/test11/'
    directory = setup_directory(base_dir, n_repeats)
    os.chdir(directory)

    save_notebook("/content/drive/My Drive/Colab Notebooks/", "Untitled8.ipynb", directory)

    # Load data and initialize autoencoder (not provided in the original code)
    # X, y = load_data()
    # ae = Autoencoder()
    X, y = generate_mock_data(50)

    labels = ['quitters', 'nonquitters']
    final_results, result = nested_cross_validation(X, y, n_repeats, random_states, ae, labels, directory)

    import os
    import json
    # Assuming you have multiple data structures to save
    data_to_save = {
  #  "shap_values_per_cv": shap_values_per_cv,
    "final_results": final_results,
    "results":result}
    data_dump(data_to_save, directory)



    # Further analysis and plotting can be added here

if __name__ == "__main__":
    main()


# ## **nbconvert and push to git**

# In[22]:


get_ipython().system('git credential-cache exit')
get_ipython().system('git config --global --unset credential.helper')
get_ipython().system('git config --global --unset user.name')
get_ipython().system('git config --global --unset user.email')


# In[21]:


get_ipython().system('rm -rf /root/.git-credentials')
get_ipython().system('rm -rf /content/.git-credentials')
get_ipython().system('rm -f /root/.git-credentials')
get_ipython().system('rm -f /root/.gitconfig')


# In[ ]:


get_ipython().system('pip install nbconvert')


# In[ ]:


from google.colab import drive
drive.mount('/content/drive/')


# In[ ]:


import os
base_dir = '/content/drive/My Drive/Colab Notebooks'
os.chdir(base_dir)


# In[18]:


#!jupyter nbconvert --to python cleaned-github-untitled8.ipynb
get_ipython().system('jupyter nbconvert --to python cleaned-github-untitled8.ipynb --output ml-pipeline-classification.py')


# In[19]:


get_ipython().system('git init')


# In[23]:


get_ipython().system('git config --global user.name "higgteil"')
get_ipython().system('git config --global user.email "pablo.reinhardt@charite.de"')


# In[24]:


get_ipython().system('git remote add origin https://higgteil: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@github.com/higgteil/eeg-predict/machine-learning')


# In[ ]:


# Set the branch (if needed)
#!git branch -M main

# Add the file
get_ipython().system('git add my_python_script.py')

# Commit the changes
get_ipython().system('git commit -m "Added data processing script"')

# Push the changes
get_ipython().system('git push -u origin main')


# In[25]:


get_ipython().system('git add ml-pipeline-classification.py')


# In[26]:


get_ipython().system('git commit -m "converted nb to py"')


# In[27]:


get_ipython().system('git push -u main')


# In[28]:


get_ipython().system('git remote -v')


# In[29]:


get_ipython().system('git remote set-url origin https://higgteil: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@github.com/higgteil/eeg-predict/machine-learning')


# In[32]:


get_ipython().system('git push origin master')


# ###troubleshooting

# In[54]:


get_ipython().system('git branch')


# In[33]:


get_ipython().system('git checkout -b main')


# In[36]:


get_ipython().system('git reset')


# In[37]:


get_ipython().system('git rm --cached structure.gdoc')


# In[53]:


get_ipython().system('echo "*.gdoc" >> .gitignore')
get_ipython().system('git add .gitignore')


# In[39]:


get_ipython().system('git restore classification_of_smoking_modularized.ipynb')
get_ipython().system('git restore stack_shap.ipynb')


# In[40]:


get_ipython().system('git add ml-pipeline-classification.py')


# In[41]:


get_ipython().system('git push -u origin main')


# In[ ]:


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


# In[42]:


get_ipython().system('git config --global user.name "higgteil"')
get_ipython().system('git config --global user.email "pablo.reinhardt@charite.de"')


# In[57]:


get_ipython().system('git remote set-url origin https://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@github.com/higgteil/eeg-predict/machine-learning.git')


# In[55]:


#!git remote set-url origin https://github.com/higgteil/eeg-predict/tree/67b376e469a94b2c630f3d2d7c225f75bdcb8993/machine_learning.git


# In[58]:


get_ipython().system('git config --global credential.helper store')
get_ipython().system('echo "https://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:x-oauth-basic@github.com" > ~/.git-credentials')


# In[59]:


get_ipython().system('git push -u origin main')

