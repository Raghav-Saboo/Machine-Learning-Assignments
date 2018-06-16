import numpy as np
import pandas as pd
import copy
from sklearn.manifold import mds
from sklearn.manifold import Isomap
from sklearn.manifold import SpectralEmbedding
from sklearn import datasets
import scipy.io
from scipy.linalg import eigh
from scipy.spatial.distance import euclidean
import math
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import functools
from mpl_toolkits.mplot3d import Axes3D
import numpy.linalg as lin
from sklearn.neighbors import NearestNeighbors
from scipy.sparse.csgraph import laplacian

def cmp(a,b):
    return a[0]-b[0]

def eigen_val(X,k):

    eigen_value, eigen_vector = lin.eig(X)
    eigen_value=np.real(eigen_value)
    eigen_vector=np.real(eigen_vector)

    print('***',eigen_value[0:5])

    #eigen_vector=abs(eigen_vector)

    eposval = []
    for i in range(len(eigen_value)):
        eposval.append([eigen_value[i],i])

    eposval.sort(key=functools.cmp_to_key(cmp))
    print('***',eposval[0:5])
    id=[]
    for i in range(1,k+1,1):
        id.append(eposval[i][1])

    print(id)

    print(eigen_value[id[0]],eigen_value[id[1]])
    ev=eigen_vector[:,id]
    return ev



def laplacain_own(X,no_of_dim,neighbour):
    n=len(X)
    weight = np.zeros([n, n])

    for i in range(len(X)):
        ndrs=NearestNeighbors(n_neighbors=neighbour,algorithm='ball_tree',metric='euclidean').fit(X)
        dis,ind=ndrs.kneighbors([[X[i][0],X[i][1],X[i][2]]])
        for j in range(len(ind[0])):
            weight[i][ind[0][j]]=1
            weight[ind[0][j]][i]=1

    r=np.zeros([n,n])
    for i in range(n):
        c=0
        for j in weight[i]:
            c=c+j
        r[i][i]=c

    l=r-weight


    #l=np.dot(np.linalg.inv(r),l)
    return eigen_val(l,no_of_dim)


n_points=100;
X, color = datasets.samples_generator.make_s_curve(n_points, random_state=0)
nx=laplacain_own(X,2,10)
#print(X)


lle=SpectralEmbedding(n_neighbors=10,n_components=2)
print(lle)
nx1=lle.fit_transform(X)
print(nx1)

'''fig = plt.figure(figsize=(15, 8))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=color, cmap=plt.cm.Spectral)
ax.view_init(12, -72)
plt.show()'''






print(nx.shape)

fig = plt.figure(figsize=(15, 8))
ax = fig.add_subplot(111)
ax.scatter(nx[:, 0], nx[:, 1],c=color, cmap=plt.cm.Spectral)
plt.show()

fig = plt.figure(figsize=(15, 8))
ax = fig.add_subplot(111)
ax.scatter(nx1[:, 0], nx1[:, 1],c=color, cmap=plt.cm.Spectral)
plt.show()







