import numpy as np
import pandas as pd
import copy
from sklearn.manifold import mds
from sklearn.manifold import Isomap
from sklearn import datasets
import scipy.io
from scipy.spatial.distance import euclidean
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy.linalg as lin
from sklearn.neighbors import NearestNeighbors

k=2


def cmp(a,b):
    return b[0]-a[0]


def mds_own(dis):
    dis = np.array(dis)

    n = len(dis)

    e = np.ones([n, 1])

    h = np.eye(n) - np.dot(e, e.T) / n

    K = np.dot(h, np.dot(dis ** 2, h))

    K = -0.5 * K

    eigen_value, eigen_vector = lin.eigh(K)
    print(eigen_value)

    '''eval=[]
    for i in range(len(eigen_value)):
        eval.append([eigen_value[i],i])

    eval.sort(key=functools.cmp_to_key(cmp))
    print(eval)

    evec=[]'''

    eposval = []
    id = []
    for i in range(len(eigen_value) - 1, 0, -1):
        if eigen_value[i] < 0:
            break
        eposval.append(eigen_value[i])
        id.append(i)

    evec = eigen_vector[:, id]
    eposval = np.array(eposval)
    n = len(eposval)
    print('Postitive ', n)

    lamb = np.eye(n)
    for i in range(len(lamb)):
        lamb[i][i] = eposval[i]

    lamb = np.sqrt(lamb)
    new_lamb = np.eye(k)

    for i in range(k):
        new_lamb[i][i] = lamb[i][i]

    v = evec[:, 0:k]

    print('Top ', k, 'eigen values')
    print(eposval[0:k])
    print()
    print()
    print('Top ', k, 'eigen vectors')
    print(v.T)
    print()
    print()
    y = np.dot(new_lamb, v.T)
    print('Final Values')
    y = y.T
    print(y)
    print()
    return y


#dis=[[0,206,429,1504,963,2976,3095,2979,1949],[206,0,233,1308,802,2815,2934,2786,1771],[429,233,0,1075,671,2684,2799,2631,1616],[1504,1308,1075,0,1329,3273,3053,2687,2037],[963,802,671,1329,0,2013,2142,2054,996],[2976,2815,2684,3273,2013,0,808,1131,1307],[3095,2934,2799,3053,2142,808,0,379,1235],[2979,2786,2631,2687,2054,1131,379,0,1059],[1949,1771,1616,2037,996,1307,1235,1059,0]]

#mds_own(dis)




n_points=400;
dis=np.zeros([n_points,n_points])
X, color = datasets.samples_generator.make_s_curve(n_points, random_state=0)
print(X)


fig = plt.figure(figsize=(15, 8))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=color, cmap=plt.cm.Spectral)
ax.view_init(12, -72)
plt.show()

for i in range(n_points):
    for j in range(n_points):
        dis[i][j]=euclidean(X[i],X[j])



nx=mds_own(dis)
dr=mds.MDS(n_components=2)
nx1=dr.fit_transform(X)


#nx=dr.fit_transform(X)
#print(nx)'''
'''dis=np.eye(len(X))
for i in range(len(X)):
    for j in range(len(X)):
        dis[i][j]=euclidean(X[i],X[j])

nx=mds_own(dis,2)'''

fig = plt.figure(figsize=(15, 8))
ax = fig.add_subplot(111)
ax.scatter(nx[:, 0], nx[:, 1],c=color, cmap=plt.cm.Spectral)
plt.show()

fig = plt.figure(figsize=(15, 8))
ax = fig.add_subplot(111)
ax.scatter(nx1[:, 0], nx1[:, 1],c=color, cmap=plt.cm.Spectral)
#plt.show()







