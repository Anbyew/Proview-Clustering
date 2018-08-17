import pandas as pd
import numpy as np
from sklearn.mixture import GaussianMixture
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
from scipy import linalg
import itertools
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
%matplotlib inline

df1 = pd.read_csv('Mid_Data_1_full.csv')
X = df1.drop(columns=['MPOG_CASE_ID','CPT', 'CPT_Predicted','Unnamed: 0']).values
print(X.shape)

spsz = 100000
indices = np.random.choice(np.arange(len(X)), size=(spsz,), replace=False)
#reduced_data = PCA(n_components=5).fit_transform(X[indices])

lowest_bic = np.infty
lowest_aic = np.infty
bic = []
aic = []
n_components_range = range(2, 60)
for n_components in n_components_range:
    gmm = GaussianMixture(n_components=n_components).fit(X[indices])
    bic.append(gmm.bic(X[indices]))
    aic.append(gmm.aic(X[indices]))
    if bic[-1] < lowest_bic:
        lowest_bic = bic[-1]
        best_gmm = gmm
    if aic[-1] < lowest_aic:
        lowest_aic = aic[-1]
        best_gmm_a = gmm

bic = np.array(bic)
bars = []
spl = plt.subplot(2, 1, 1)
xpos = np.array(n_components_range) + .2 * (-2)
bars.append(plt.bar(xpos, bic[0:len(n_components_range)]))

aic = np.array(aic)
bars = []
spl = plt.subplot(2, 1, 2)
xpos = np.array(n_components_range) + .2 * (-2)
bars.append(plt.bar(xpos, aic[0:len(n_components_range)]))

print(best_gmm)
print(best_gmm_a)


splot = plt.subplot(2, 1, 2)
Y_ = best_gmm.predict(X[indices])
visu = PCA(n_components=2).fit_transform(X[indices])
for i, (mean, cov, color) in enumerate(zip(best_gmm.means_, best_gmm.covariances_,
                                           color_iter)):
    v, w = linalg.eigh(cov)
    if not np.any(Y_ == i):
        continue
    plt.scatter(visu[Y_ == i, 0], visu[Y_ == i, 1], .8, color=color)

    # Plot an ellipse to show the Gaussian component
    angle = np.arctan2(w[0][1], w[0][0])
    angle = 180. * angle / np.pi  # convert to degrees
    v = 2. * np.sqrt(2.) * np.sqrt(v)
    ell = mpl.patches.Ellipse(mean, v[0], v[1], 180. + angle, color=color)
    ell.set_clip_box(splot.bbox)
    ell.set_alpha(.5)
    splot.add_artist(ell)





