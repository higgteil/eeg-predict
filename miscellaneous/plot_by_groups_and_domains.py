# -*- coding: utf-8 -*-
"""
# input: lists / dataframes
# output: grouped [by class and category] barplot with errorbars 
"""

import os
import pandas as pd 
import glob 


import matplotlib as mpl
from matplotlib.patches import Patch
from matplotlib import lines
from matplotlib.lines import Line2D

directory = "/content/drive/MyDrive/plots"

cols = ["sociodemographic-\nenvironmental","clinical","neuropsychological","EEG"]

## qnq 
list_row_ROC_mean_qnq = [0.69, 0.60, 0.50, 0.51]
list_row_ROC_sd_qnq =   [0.14,0.13,0.15,0.17]

## sns 
list_row_ROC_mean_sns=[0.78, 0.68, 0.62, 0.70]
list_row_ROC_sd_sns = [0.04, 0.03, 0.04, 0.05]

## hsns 
list_row_ROC_mean_hsns=[0.84, 0.71, 0.70, 0.83]
list_row_ROC_sd_hsns = [0.05, 0.05, 0.06, 0.04]

## hsls
list_row_ROC_mean_hsls=[0.65, 0.60, 0.57, 0.66]
list_row_ROC_sd_hsls = [0.06, 0.06, 0.08, 0.06]

# 3 groups with HSLS
df3 = pd.DataFrame({
    'group': ["sns","sns","sns","sns",
              "hsns","hsns","hsns","hsns",
              "qnq","qnq","qnq","qnq"],
# 4 groups with HSLS
df4 = pd.DataFrame({
    'group': ["sns","sns","sns","sns",
              "hsns","hsns","hsns","hsns",
              "qnq","qnq","qnq","qnq",
              "hsls","hsls","hsls","hsls"],
    'mean': list_row_ROC_mean_sns+list_row_ROC_mean_hsns+list_row_ROC_mean_qnq+list_row_ROC_mean_hsls,
    'domain':cols+cols+cols+cols,
    'std':list_row_ROC_sd_sns+list_row_ROC_sd_hsns+ list_row_ROC_sd_qnq+ list_row_ROC_sd_hsls})

mpl.rcParams.update(mpl.rcParamsDefault)
fig = plt.figure(dpi=400)
SMALL_SIZE = 12
MEDIUM_SIZE = 14
BIGGER_SIZE = 16

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title


colors = ["RdBu","seismic","Dark2"]

files = [df3, df4]

groups = ["core_groups","all_groups"]
for f,gr in zip(files,groups):
    vals = f.pivot(index='group', columns='domain', values='mean')
    yerr = f.pivot(index='group', columns="domain", values='std')

    # plot vals with yerr
    ax = vals.plot(kind='bar', yerr=yerr, logy=False, rot=30,colormap=colors[0],figsize=(10, 7))
    plt.title("ROC AUC per domain by classes",fontweight='bold')
    plt.ylabel("ROC AUC",fontweight='bold')
    plt.xlabel("group",fontweight='bold')
    legend_elements = [Line2D([0], [0], color='black', lw=4, label='std')]
    # Create the figure
    legend = ax.legend(handles=legend_elements, bbox_to_anchor=(1, 1.02), loc='upper left')
    ax = legend.axes
    handles, labels = ax.get_legend_handles_labels()
    handles.append(lines.Line2D([1,5],[1,2], color = 'black', label = 'Vertical line'))
    labels.append("std")
    legend._legend_box = None
    legend._init_legend_box(handles, labels)
    legend._set_loc(legend._loc)
    legend.set_title(legend.get_title().get_text())
    if not os.path.exists(directory):
           os.makedirs(directory)
    else: os.chdir(directory)
    #ax.legend(title='domains', bbox_to_anchor=(1, 1.02), loc='upper left')
    plt.tight_layout()
    title = "ROC_by_domain__{}_{}.png".format(colors[0],gr)
   # plt.savefig(title,layout="tight")
    plt.show()
    plt.close("all")
