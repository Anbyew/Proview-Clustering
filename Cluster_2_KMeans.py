import pandas as pd
import numpy as np

from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from scipy.spatial import distance
from collections import Counter 
import matplotlib.pyplot as plt 
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D

from sklearn.metrics import silhouette_score
from sklearn.metrics import calinski_harabaz_score
from sklearn.metrics import pairwise_distances

%matplotlib inline 

df1 = pd.read_csv('Mid_Data_2_full.csv')
X = df1.drop(columns=['Unnamed: 0']).values
print(X.shape)

spsz = 10000
indices = np.random.choice(np.arange(len(X)), size=(spsz,), replace=False)
reduced_data = PCA(n_components=5).fit_transform(X[indices])

ss = []
cs = []
elb = []
for cn in range(2,30):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    kmeans = KMeans(n_clusters=cn).fit(reduced_data)
    ts = silhouette_score(X[indices], kmeans.labels_)
    tc = calinski_harabaz_score(X[indices], kmeans.labels_)
    eb = kmeans.inertia_
    ss.append(ts)
    cs.append(tc)
    elb.append(eb)

    kmtd = PCA(n_components=3).fit_transform(reduced_data)
    #kmtd = TSNE(n_components=3).fit_transform(reduced_data)

    print("###############")
    print("cn: " + str(cn))
    print("ts: " + str(ts))
    print("tc: " + str(tc))
    print("eb: " + str(eb))

    for i in range(kmtd.shape[0]):
        ax.scatter(kmtd[i, 0], kmtd[i, 1], kmtd[i,2],color=plt.cm.nipy_spectral(kmeans.labels_[i]/cn))
    plt.savefig('1/kmeans_' + str(cn) + '_' + str(spsz) + '.png')
    print("")



plt.plot(range(2,30), ss)
plt.savefig('1/kmeans_' + str(cn) + '_' + str(spsz) + '_ss.png')
plt.plot(range(2,30), cs)
plt.savefig('1/kmeans_' + str(cn) + '_' + str(spsz) + '_cs.png')
plt.plot(range(2,30), elb)
plt.savefig('1/kmeans_' + str(cn) + '_' + str(spsz) + '_elb.png')