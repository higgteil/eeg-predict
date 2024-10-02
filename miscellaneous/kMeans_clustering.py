# -*- coding: utf-8 -*-

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

from sklearn.decomposition import PCA


##################################################################################################################
# first, decide how many clusters by the elbow criteria 
#
##################################################################################################################
measure='xxyyzzelbow_PCA'
pca = PCA(n_components=2)


Sum_of_squared_distances = []
K = range(1,15)
for k in K:
    X_pca = pca.fit_transform(X_)
    km = KMeans(n_clusters=k)
    km = km.fit(X_pca)
    Sum_of_squared_distances.append(km.inertia_)

plt.plot(K, Sum_of_squared_distances, 'bx-')
plt.xlabel('k')
plt.ylabel('Sum_of_squared_distances')
plt.title('Elbow Method For Optimal k: '+measure)

directory = "/content/drive/MyDrive/plot_PCA_embedded_kMeans/"
import os
if not os.path.exists(directory):
    os.makedirs(directory)
else: os.chdir(directory)
plt.tight_layout()
title = "figure_KMeans_elbow_"+measure+".png"
plt.savefig(str(title))
plt.show()





# preprocess
impute = SimpleImputer(strategy="most_frequent")
scale = MinMaxScaler()
custom_pipeline = make_pipeline(scale, impute)
# scale 
X_r = custom_pipeline.fit_transform(X)
X_r = pd.DataFrame(X_r, columns=X.columns)

# 
measure ="xxyyzz"


km = KMeans(
    n_clusters=4, init='random',
    n_init=10, max_iter=300, 
    tol=1e-04, random_state=20
)



# fit PCA on MinMax scaled & filtered only smokers X 
X_pca = PCA(n_components=2).fit_transform(X_r)

# fit k-Means Clustering on PCA-embedded filtered only smokers X 
y_km = km.fit_predict(X_pca)




# dpi 
fig = plt.figure(figsize=(8,8), dpi=100)



# plot the 3 clusters
plt.scatter(
    X_pca[y_km == 0, 0], X_pca[y_km == 0, 1],
    s=50, c='purple',
    marker='s', edgecolor='black',
    label='cluster 1'
)

plt.scatter(
    X_pca[y_km == 1, 0], X_pca[y_km == 1, 1],
    s=50, c='orange',
    marker='o', edgecolor='black',
    label='cluster 2'
)

plt.scatter(
    X_pca[y_km == 2, 0], X_pca[y_km == 2, 1],
    s=50, c='lightblue',
    marker='v', edgecolor='black',
    label='cluster 3'
)
plt.scatter(
    X_pca[y_km == 3, 0], X_pca[y_km == 3, 1],
    s=50, c='cornflowerblue',
    marker='h', edgecolor='black',
    label='cluster 4'
)


#plt.scatter(
#    X_pca[y_km == 4, 0], X_pca[y_km == 4, 1],
#    s=50, c='cornsilk',
#    marker='D', edgecolor='black',
#    label='cluster 5'
#)


#plt.scatter(
#    X_pca[y_km == 5, 0], X_pca[y_km == 5, 1],
#    s=50, c='indianred',
#    marker=9, edgecolor='black',
#    label='cluster 6'
#)


#plt.scatter(
#    X_pca[y_km == 5, 0], X_pca[y_km == 5, 1],
#    s=50, c='firebrick',
#    marker=11, edgecolor='black',
#    label='cluster 7'
#)


#plt.scatter(
#    X_pca[y_km == 5, 0], X_pca[y_km == 5, 1],
#    s=50, c='aliceblue',
#    marker=5, edgecolor='black',
#    label='cluster 8'
#)


# plot the centroids
plt.scatter(
    km.cluster_centers_[:, 0], km.cluster_centers_[:, 1],
    s=250, marker='*',
    c='red', edgecolor='black',
    label='centroids'
)


plt.legend(scatterpoints=1)
plt.grid()
title = "PCA (k=2) embedded k-Means Clustering (k=4)\n "+measure
plt.title(title,fontsize=14)
plt.xlabel("Clusters", fontsize=14)


directory = "/content/drive/MyDrive/plot_kMeans/"
import os
if not os.path.exists(directory):
    os.makedirs(directory)
else: os.chdir(directory)


plt.tight_layout()
title = "figure_PCA_"+measure+".png"
plt.savefig(str(title))


plt.show()
plt.close()
