import pandas as pd
import numpy as np
from sklearn.cluster import SpectralClustering
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

spsz = 10000
indices = np.random.choice(np.arange(len(X)), size=(spsz,), replace=False)
reduced_data = PCA(n_components=5).fit_transform(X[indices])

ss = []
cs = []
for cn in range(2,30):
    print("##############################")
    print("number of clusters: " + str(cn))
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    X[indices] = StandardScaler().fit_transform(X[indices]) #doesn't work with LLE: singular matrix
    
    spectral = SpectralClustering(n_clusters=cn).fit(reduced_data)
    labels = spectral.labels_
    ts = silhouette_score(X[indices], labels)
    tc = calinski_harabaz_score(X[indices], labels)
    ss.append(ts)
    cs.append(tc)
    print("ts: " + str(ts))
    print("tc: " + str(tc))
    
    sptd = PCA(n_components=3).fit_transform(reduced_data)

    for i in range(sptd.shape[0]):
        ax.scatter(sptd[i, 0], sptd[i, 1], sptd[i,2],color=plt.cm.nipy_spectral(spectral.labels_[i]/cn))
    plt.savefig('1/spectral_' + str(cn) + '_' + str(spsz) + '.png')
    print("")

plt.plot(range(2,30), ss)
plt.savefig('1/spectral_' + str(cn) + '_' + str(spsz) + '_ss.png')
plt.plot(range(2,30), cs)
plt.savefig('1/spectral_' + str(cn) + '_' + str(spsz) + '_cs.png')
