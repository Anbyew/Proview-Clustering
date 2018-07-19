# Proview Medicine Clustering
# Clustering method: DBScan
# Dimension Reduction methods: PCA, LLE, SE, MDS, TSNE
#
# Date: July 17, 2018
# Author: Yuwei Bao(yuweibao@umich.edu)
# Sponcered by: Department of Anesthesiology, University of Michigan


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
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm

df3 = pd.read_csv('Meds_pros_aws.csv')
X = df3.drop(columns=['Unnamed: 0']).values
print(X.shape)


epp = [0.35, 0.40, 0.45, 0.48, 0.5]
mii = [3, 5, 7]


def dbplt(reduced_data, ep, mi, meth):
	db = DBSCAN(eps=ep, min_samples=mi).fit(reduced_data)
	core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
	core_samples_mask[db.core_sample_indices_] = True
	labels = db.labels_

	n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
	print('Estimated number of clusters: %d' % n_clusters_)

	if n_clusters_ < 120:
		print("Ploting...")

		fig = plt.figure()
		ax = fig.add_subplot(111, projection='3d')
		unique_labels = set(labels)
		colors = iter(cm.rainbow(np.linspace(0, 1, len(unique_labels))))

		for k, col in zip(unique_labels, colors):
			if k == -1:
				col = [0, 0, 0, 1]

			class_member_mask = (labels == k)
			xy = reduced_data[class_member_mask & core_samples_mask]
			ax.scatter(xy[:, 0], xy[:, 1], xy[:, 2], 'o', c=tuple(col), s = 10)


			xy = reduced_data[class_member_mask & ~core_samples_mask]
			ax.scatter(xy[:, 0], xy[:, 1], xy[:, 2], 'o', c=tuple(col), s = 4)


		plt.savefig('DB_10000_3dplt_' + str(meth) + '_'+ str(ep) + '_' + str(mi) + '_' + str(n_clusters_) + '.png')
		print("")


for ep in epp:
	for mi in mii:
		print("############################")
		print("ep: " + str(ep))
		print("mi: " + str(mi))
		indices = np.random.choice(np.arange(len(X)), size=(10000,), replace=False)
		#X = StandardScaler().fit_transform(X)
		
		print("@@@@@@PCA~~~~~")
		reduced_data = PCA(n_components=3).fit_transform(X[indices])
		dbplt(reduced_data, ep, mi, 'PCA')
		
		print("@@@@@@LLE~~~~~")
		reduced_data = LocallyLinearEmbedding(n_components=3).fit_transform(X[indices])
		dbplt(reduced_data, ep, mi, 'LLE')
		
		print("@@@@@@SE~~~~~")
		reduced_data = SpectralEmbedding(n_components=3).fit_transform(X[indices])
		dbplt(reduced_data, ep, mi, 'SE')
		
		print("@@@@@@TSNE~~~~~")
		reduced_data = TSNE(n_components=3).fit_transform(X[indices])
		dbplt(reduced_data, ep, mi, 'TSNE')

		print("@@@@@@MDS~~~~~")
		reduced_data = MDS(n_components=3).fit_transform(X[indices])
		dbplt(reduced_data, ep, mi, 'MDS')
		
		
		print("********END OF THIS ROUNT**********")
		print("")