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



def mds_own(dis,k):
    dis = np.array(dis)

    n = len(dis)

    e = np.ones([n, 1])

    h = np.eye(n) - np.dot(e, e.T) / n

    K = np.dot(h, np.dot(dis**2, h))

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

def isomap_own(X,no_of_dim,neighbour):
    n=len(X)
    adj = 1e10*np.ones([n, n])

    for i in range(len(adj)):
        adj[i][i]=0

    for i in range(len(X)):
        ndrs=NearestNeighbors(n_neighbors=neighbour,algorithm='ball_tree',metric='euclidean').fit(X)
        dis,ind=ndrs.kneighbors([[X[i][0],X[i][1],X[i][2]]])
        print(dis,ind)
        print('**',len(dis))
        for j in range(len(ind[0])):
            adj[i][ind[0][j]]=dis[0][j]
            adj[ind[0][j]][i]=dis[0][j]

    print('****')
    print(len(adj[0]))
    print(adj[0])

    dis=copy.deepcopy(adj)
    n=len(adj)
    for k in range(n):
        for i in range(n):
            for j in range(n):
                if dis[i][j]>dis[i][k]+dis[k][j]:
                    dis[i][j]=dis[i][k]+dis[k][j]




    dr = mds.MDS(n_components=2, dissimilarity="precomputed")
    print(dr.fit_transform(dis))
    y = dr.fit_transform(dis)
    #return y
    return mds_own(dis,no_of_dim)

'''fp=np.loadtxt('foo.txt',delimiter=',')
print(fp[:,0:5])

data=np.array(fp)

dr=Isomap(n_components=2,n_neighbors=3)
print(dr)
X=dr.fit_transform(data)
print(X)'''


n_points=200;
X, color = datasets.samples_generator.make_s_curve(n_points, random_state=0)
nx=isomap_own(X,2,10)
print(X)


'''fig = plt.figure(figsize=(15, 8))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=color, cmap=plt.cm.Spectral)
ax.view_init(12, -72)
plt.show()'''



dr=Isomap(n_components=2,n_neighbors=10)
#dr=mds.MDS(n_components=2)
nx1=dr.fit_transform(X)
print(dr)


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
plt.show()







