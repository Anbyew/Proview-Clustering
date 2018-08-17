import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.manifold import MDS
from sklearn.manifold import TSNE
from sklearn.manifold import SpectralEmbedding
from sklearn.manifold import LocallyLinearEmbedding
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.metrics import calinski_harabaz_score
from sklearn.metrics import pairwise_distances
from sklearn import datasets
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm
%matplotlib inline 

df1 = pd.read_csv('Mid_Data_1_full.csv')
X = df1.drop(columns=['MPOG_CASE_ID','CPT', 'CPT_Predicted','Unnamed: 0']).values
print(X.shape)

spsz = 20000
indices = np.random.choice(np.arange(len(X)), size=(spsz,), replace=False)
reduced_data = PCA(n_components=5).fit_transform(X[indices])

epp = [0.3, 0.35, 0.4, 0.45, 0.5, 0.55]
mii = [5, 6, 10, 12]
ss = []
cs = []
for ep in epp:
    for mi in mii:
        print("############################")
        print("ep: " + str(ep))
        print("mi: " + str(mi))
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        db = DBSCAN(eps=ep, min_samples=mi).fit(reduced_data)
        core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
        core_samples_mask[db.core_sample_indices_] = True
        labels = db.labels_
        ts = silhouette_score(X[indices], labels)
        tc = calinski_harabaz_score(X[indices], labels)
        ss.append(ts)
        cs.append(tc)
        print("ts: " + str(ts))
        print("tc: " + str(tc))

        n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

        if n_clusters_ < 120:
            print('Estimated number of clusters: %d' % n_clusters_)

            #kmtsne = TSNE(n_components=3).fit_transform(X[indices])
            kmtsne = PCA(n_components=3).fit_transform(reduced_data)

            unique_labels = set(labels)
            colors = iter(cm.rainbow(np.linspace(0, 1, len(unique_labels))))

            for k, col in zip(unique_labels, colors):
                if k == -1:
                    col = [0, 0, 0, 1]

                class_member_mask = (labels == k)
                xy = kmtsne[class_member_mask & core_samples_mask]
                ax.scatter(xy[:, 0], xy[:, 1], xy[:, 2], 'o', c=tuple(col), s = 10)


                xy = kmtsne[class_member_mask & ~core_samples_mask]
                ax.scatter(xy[:, 0], xy[:, 1], xy[:, 2], 'o', c=tuple(col), s = 4)

            plt.savefig('1/DBSCAN' + str(ep) + '_' + str(mi) + '_' + str(spsz) + '.png')
            print("")



plt.plot(range(len(epp) * len(mii)), ss)
plt.savefig('1/DBSCAN' + str(ep) + '_' + str(mi) + '_' + str(spsz) + '_ss.png')
plt.plot(range(len(epp) * len(mii)), cs)
plt.savefig('1/DBSCAN' + str(ep) + '_' + str(mi) + '_' + str(spsz) + '_cs.png')           