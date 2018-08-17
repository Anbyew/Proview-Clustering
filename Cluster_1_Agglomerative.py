import pandas as pd
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.neighbors import kneighbors_graph
from sklearn.manifold import TSNE
from sklearn.manifold import MDS
from sklearn.manifold import LocallyLinearEmbedding
from sklearn.manifold import SpectralEmbedding
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.metrics import calinski_harabaz_score
from sklearn.metrics import pairwise_distances
from sklearn import datasets
from collections import Counter 
import matplotlib.pyplot as plt 
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D
%matplotlib inline

df1 = pd.read_csv('Mid_Data_1_full.csv')
X = df1.drop(columns=['MPOG_CASE_ID','CPT', 'CPT_Predicted','Unnamed: 0']).values
print(X.shape)

spsz = 100000
indices = np.random.choice(np.arange(len(X)), size=(spsz,), replace=False)
reduced_data = PCA(n_components=5).fit_transform(X[indices])

ss = []
cs = []
for cn in range(2,30):
    print("##############################")
    print("number of clusters: " + str(cn))
    X[indices] = StandardScaler().fit_transform(X[indices]) #doesn't work with LLE: singular matrix
    
    reduced_data = PCA(n_components=2).fit_transform(X[indices])  
    clustering = AgglomerativeClustering(n_clusters=cn).fit(reduced_data)
    labels = clustering.labels_
    ts = silhouette_score(X[indices], labels)
    tc = calinski_harabaz_score(X[indices], labels)
    ss.append(ts)
    cs.append(tc)
    print("ts: " + str(ts))
    print("tc: " + str(tc))
    
    x_min, x_max = np.min(reduced_data, axis=0), np.max(reduced_data, axis=0)
    reduced_data = (reduced_data - x_min) / (x_max - x_min)
    plt.figure(figsize=(6, 4))
    for i in range(reduced_data.shape[0]):
        plt.text(reduced_data[i, 0], reduced_data[i, 1], str(clustering.labels_[i]),
                color=plt.cm.nipy_spectral(clustering.labels_[i]/cn),
                 fontdict={'weight': 'bold', 'size': 6})

    plt.savefig('1/Agg_' + str(cn) + '_' + str(spsz) + '.png')    
    print("")


plt.plot(range(2,30), ss)
plt.savefig('1/Agg_' + str(cn) + '_' + str(spsz) + '_ss.png')
plt.plot(range(2,30), cs)
plt.savefig('1/Agg_' + str(cn) + '_' + str(spsz) + '_cs.png')