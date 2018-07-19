# Proview Medicine Clustering
# Clustering method: Agglomerative Clustering
# Dimension Reduction methods: PCA, LLE, SE, MDS, TSNE
#
# Date: July 17, 2018
# Author: Yuwei Bao(yuweibao@umich.edu)
# Sponcered by: Department of Anesthesiology, University of Michigan


import pandas as pd
import numpy as np

# import data
df2 = pd.read_csv('Meds_pros_aws.csv')
X = df2.drop(columns=['Unnamed: 0']).values
print(X.shape)


from sklearn.cluster import AgglomerativeClustering
from sklearn.neighbors import kneighbors_graph
from sklearn.manifold import TSNE
from sklearn.manifold import MDS
from sklearn.manifold import LocallyLinearEmbedding
from sklearn.manifold import SpectralEmbedding
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from collections import Counter 
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt 
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D


# run 5 different dimension reduction methods
nc = [4, 6, 8, 10, 12, 20, 30]
for cn in nc:
	print("##############################")
	print("number of clusters: " + str(cn))
	indices = np.random.choice(np.arange(len(X)), size=(50000,), replace=False)
	#X = StandardScaler().fit_transform(X) #doesn't work with LLE: singular matrix
	
	print("@@@@@@PCA~~~~~")
	reduced_data = PCA(n_components=2).fit_transform(X[indices])  
	clustering = AgglomerativeClustering(n_clusters=cn).fit(reduced_data)
	print("Ploting...")
	x_min, x_max = np.min(reduced_data, axis=0), np.max(reduced_data, axis=0)
	reduced_data = (reduced_data - x_min) / (x_max - x_min)
	plt.figure(figsize=(6, 4))
	for i in range(reduced_data.shape[0]):
	    plt.text(reduced_data[i, 0], reduced_data[i, 1], str(clustering.labels_[i]),
	             color=plt.cm.nipy_spectral(clustering.labels_[i]/cn),
	             fontdict={'weight': 'bold', 'size': 6})

	plt.savefig('AGG_10000_PCA_2dplt_' + str(cn) + '.png')
	print("")
	
	print("@@@@@@LLE~~~~~")
	reduced_data = LocallyLinearEmbedding(n_components=2).fit_transform(X[indices])  
	clustering = AgglomerativeClustering(n_clusters=cn).fit(reduced_data)
	print("Ploting...")
	x_min, x_max = np.min(reduced_data, axis=0), np.max(reduced_data, axis=0)
	reduced_data = (reduced_data - x_min) / (x_max - x_min)
	plt.figure(figsize=(6, 4))
	for i in range(reduced_data.shape[0]):
	    plt.text(reduced_data[i, 0], reduced_data[i, 1], str(clustering.labels_[i]),
	             color=plt.cm.nipy_spectral(clustering.labels_[i]/cn),
	             fontdict={'weight': 'bold', 'size': 6})

	plt.savefig('AGG_10000_LLE_2dplt_' + str(cn) + '.png')
	print("")
	
	
	print("@@@@@@SE~~~~~")
	reduced_data = SpectralEmbedding(n_components=2).fit_transform(X[indices])
	clustering = AgglomerativeClustering(n_clusters=cn).fit(reduced_data)
	print("Ploting...")
	x_min, x_max = np.min(reduced_data, axis=0), np.max(reduced_data, axis=0)
	reduced_data = (reduced_data - x_min) / (x_max - x_min)
	plt.figure(figsize=(6, 4))
	for i in range(reduced_data.shape[0]):
	    plt.text(reduced_data[i, 0], reduced_data[i, 1], str(clustering.labels_[i]),
	             color=plt.cm.nipy_spectral(clustering.labels_[i]/cn),
	             fontdict={'weight': 'bold', 'size': 6})

	plt.savefig('AGG_10000_SE_2dplt_' + str(cn) + '.png')
	print("")
	

	print("@@@@@@TSNE~~~~~")
	reduced_data = PCA(n_components=10).fit_transform(X[indices])
	reduced_data = TSNE(n_components=2).fit_transform(reduced_data) 
	clustering = AgglomerativeClustering(n_clusters=cn).fit(reduced_data)
	print("Ploting...")
	x_min, x_max = np.min(reduced_data, axis=0), np.max(reduced_data, axis=0)
	reduced_data = (reduced_data - x_min) / (x_max - x_min)
	plt.figure(figsize=(6, 4))
	for i in range(reduced_data.shape[0]):
		plt.text(reduced_data[i, 0], reduced_data[i, 1], str(clustering.labels_[i]),
				 color=plt.cm.nipy_spectral(clustering.labels_[i]/cn),
				 fontdict={'weight': 'bold', 'size': 6})
	plt.savefig('AGG_10000_TSNE_2dplt_' + str(cn) + '.png')


	
	print("@@@@@@MDS~~~~~")
	reduced_data = PCA(n_components=10).fit_transform(X[indices]) 
	reduced_data = MDS(n_components=2).fit_transform(reduced_data) 
	clustering = AgglomerativeClustering(n_clusters=cn).fit(reduced_data)
	print("Ploting...")
	x_min, x_max = np.min(reduced_data, axis=0), np.max(reduced_data, axis=0)
	reduced_data = (reduced_data - x_min) / (x_max - x_min)
	plt.figure(figsize=(6, 4))
	for i in range(reduced_data.shape[0]):
		plt.text(reduced_data[i, 0], reduced_data[i, 1], str(clustering.labels_[i]),
				 color=plt.cm.nipy_spectral(clustering.labels_[i]/cn),
				 fontdict={'weight': 'bold', 'size': 6})

	plt.savefig('AGG_10000_MDS_2dplt_' + str(cn) + '.png')
	print("")
	
	
	print("********END OF THIS ROUNT**********")
	print("")