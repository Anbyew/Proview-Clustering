# Proview Medicine Clustering
# Clustering method: Kmeans
# Dimension Reduction methods: PCA, LLE, SE, MDS, TSNE
#
# Date: July 17, 2018
# Author: Yuwei Bao(yuweibao@umich.edu)
# Sponcered by: Department of Anesthesiology, University of Michigan


import pandas as pd
import numpy as np


df3 = pd.read_csv('Meds_pros_aws.csv')
X = df3.drop(columns=['Unnamed: 0']).values
print(X.shape)

from sklearn.cluster import KMeans
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



nc = [4, 6, 8, 10, 12, 20, 30]
for cn in nc:
    print("##############################")
    print("number of clusters: " + str(cn))
    print("")
    indices = np.random.choice(np.arange(len(X)), size=(10000,), replace=False)
    #X = StandardScaler().fit_transform(X)
    
    
    print("@@@@@@PCA~~~~~")
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    reduced_data = PCA(n_components=3).fit_transform(X[indices])     
    kmeans = KMeans(n_clusters=cn).fit(reduced_data)
    
    cluster_dist = kmeans.transform(reduced_data)
    clusters = np.argmax(cluster_dist, axis=1)
    counter = Counter(clusters)
    print("number of counters: " + str(len(counter)))

    colors = iter(cm.rainbow(np.linspace(0, 1, len(list(counter.keys())))))
    for cl in list(counter.keys()):
        #plt.scatter(reduced_data[clusters == cl, 0], reduced_data[clusters == cl, 1], color=next(colors))
        xs = reduced_data[clusters == cl, 0]
        ys = reduced_data[clusters == cl, 1]
        zs = reduced_data[clusters == cl, 2]
        ax.scatter(xs, ys, zs, c=next(colors))
    plt.savefig('KM_10000_PCA_3dplt' + str(cn) + '.png')
    print("")
    
    
    
    
    print("@@@@@@LLE~~~~~")
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    reduced_data = LocallyLinearEmbedding(n_components=3).fit_transform(X[indices])   
    kmeans = KMeans(n_clusters=cn).fit(reduced_data)
    
    cluster_dist = kmeans.transform(reduced_data)
    clusters = np.argmax(cluster_dist, axis=1)
    counter = Counter(clusters)
    print("number of counters: " + str(len(counter)))

    colors = iter(cm.rainbow(np.linspace(0, 1, len(list(counter.keys())))))
    for cl in list(counter.keys()):
        #plt.scatter(reduced_data[clusters == cl, 0], reduced_data[clusters == cl, 1], color=next(colors))
        xs = reduced_data[clusters == cl, 0]
        ys = reduced_data[clusters == cl, 1]
        zs = reduced_data[clusters == cl, 2]
        ax.scatter(xs, ys, zs, c=next(colors))
    plt.savefig('KM_10000_LLE_3dplt' + str(cn) + '.png')
    print("")
    
    
    print("@@@@@@SE~~~~~")
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    reduced_data = SpectralEmbedding(n_components=3).fit_transform(X[indices])   
    kmeans = KMeans(n_clusters=cn).fit(reduced_data)
    
    cluster_dist = kmeans.transform(reduced_data)
    clusters = np.argmax(cluster_dist, axis=1)
    counter = Counter(clusters)
    print("number of counters: " + str(len(counter)))

    colors = iter(cm.rainbow(np.linspace(0, 1, len(list(counter.keys())))))
    for cl in list(counter.keys()):
        #plt.scatter(reduced_data[clusters == cl, 0], reduced_data[clusters == cl, 1], color=next(colors))
        xs = reduced_data[clusters == cl, 0]
        ys = reduced_data[clusters == cl, 1]
        zs = reduced_data[clusters == cl, 2]
        ax.scatter(xs, ys, zs, c=next(colors))
    plt.savefig('KM_10000_SE_3dplt' + str(cn) + '.png')
    print("")
    
    
    
    print("@@@@@@MDS~~~~~")
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    reduced_data = PCA(n_components=10).fit_transform(X[indices].astype(np.float64)) 
    reduced_data = MDS(n_components=3).fit_transform(reduced_data.astype(np.float64))   
    kmeans = KMeans(n_clusters=cn).fit(reduced_data)
    
    cluster_dist = kmeans.transform(reduced_data)
    clusters = np.argmax(cluster_dist, axis=1)
    counter = Counter(clusters)
    print("number of counters: " + str(len(counter)))

    colors = iter(cm.rainbow(np.linspace(0, 1, len(list(counter.keys())))))
    for cl in list(counter.keys()):
        #plt.scatter(reduced_data[clusters == cl, 0], reduced_data[clusters == cl, 1], color=next(colors))
        xs = reduced_data[clusters == cl, 0]
        ys = reduced_data[clusters == cl, 1]
        zs = reduced_data[clusters == cl, 2]
        ax.scatter(xs, ys, zs, c=next(colors))
    plt.savefig('KM_10000_MDS_3dplt' + str(cn) + '.png')
    print("")
    
    
    
    print("@@@@@@TSNE~~~~~")
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    reduced_data = PCA(n_components=10).fit_transform(X[indices].astype(np.float64))
    reduced_data = TSNE(n_components=3).fit_transform(reduced_data.astype(np.float64))   
    kmeans = KMeans(n_clusters=cn).fit(reduced_data)
    
    cluster_dist = kmeans.transform(reduced_data)
    clusters = np.argmax(cluster_dist, axis=1)
    counter = Counter(clusters)
    print("number of counters: " + str(len(counter)))

    colors = iter(cm.rainbow(np.linspace(0, 1, len(list(counter.keys())))))
    for cl in list(counter.keys()):
        #plt.scatter(reduced_data[clusters == cl, 0], reduced_data[clusters == cl, 1], color=next(colors))
        xs = reduced_data[clusters == cl, 0]
        ys = reduced_data[clusters == cl, 1]
        zs = reduced_data[clusters == cl, 2]
        ax.scatter(xs, ys, zs, c=next(colors))
    plt.savefig('KM_10000_TSNE_3dplt' + str(cn) + '.png')
    print("********END OF THIS ROUNT**********")
    print("")