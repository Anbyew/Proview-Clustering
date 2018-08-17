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

df1 = pd.read_csv('Mid_Data_1_full.csv')
X = df1.drop(columns=['MPOG_CASE_ID','CPT', 'CPT_Predicted','Unnamed: 0']).values
print(X.shape)

spsz = 100
indices = np.random.choice(np.arange(len(X)), size=(spsz,), replace=False)
reduced_data = PCA(n_components=5).fit_transform(X[indices])

def compute_bic(kmeans,X):
    # assign centers and labels
    centers = [kmeans.cluster_centers_]
    labels  = kmeans.labels_
    #number of clusters
    m = kmeans.n_clusters
    # size of the clusters
    n = np.bincount(labels)
    #size of data set
    N, d = X.shape

    #compute variance for all clusters beforehand
    cl_var = (1.0 / (N - m) / d) * sum([sum(distance.cdist(X[np.where(labels == i)], [centers[0][i]], 
             'euclidean')**2) for i in range(m)])

    const_term = 0.5 * m * np.log(N) * (d+1)

    BIC = np.sum([n[i] * np.log(n[i]) -
               n[i] * np.log(N) -
             ((n[i] * d) / 2) * np.log(2*np.pi*cl_var) -
             ((n[i] - 1) * d/ 2) for i in range(m)]) - const_term

    return(BIC)


ss = []
cs = []
elb = []
bic = []
for cn in range(2,30):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    kmeans = KMeans(n_clusters=cn).fit(reduced_data)
    ts = silhouette_score(X[indices], kmeans.labels_)
    tc = calinski_harabaz_score(X[indices], kmeans.labels_)
    eb = kmeans.inertia_
    #tic = compute_bic(kmeans,X[indices]) 
    ss.append(ts)
    cs.append(tc)
    elb.append(eb)
    #bic.append(tic)

    kmtd = PCA(n_components=3).fit_transform(reduced_data)
    #kmtd = TSNE(n_components=3).fit_transform(reduced_data)

    print("###############")
    print("cn: " + str(cn))
    print("ts: " + str(ts))
    print("tc: " + str(tc))
    print("eb: " + str(eb))
    #print("tic: " + str(tic))

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
# plt.plot(range(2,30), bic)
# plt.savefig('1/kmeans_' + str(cn) + '_' + str(spsz) + '_bic.png')
