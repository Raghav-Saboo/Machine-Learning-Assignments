from sklearn.neighbors import NearestNeighbors
from sklearn import datasets, neighbors
import numpy as np
from scipy import linalg
from scipy.linalg import eigh
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm


# we will implement K-nearest neighbor search
def Knbor_Mat(X, K, t=2.0, dist_metric="euclidean", algorithm="ball_tree"):
    n, p = X.shape

    knn = neighbors.NearestNeighbors(K + 1, metric=dist_metric, algorithm=algorithm).fit(X)
    distances, nbors = knn.kneighbors(X)

    return (nbors[:, 1:])


def get_weights(X, nbors, reg, K):
    n, p = X.shape

    Weights = np.zeros((n, n))

    for i in range(n):

        X_bors = X[nbors[i], :] - X[i]
        cov_nbors = np.dot(X_bors, X_bors.T)

        # regularization tems
        trace = np.trace(cov_nbors)
        if trace > 0:
            R = reg * trace
        else:
            R = reg

        cov_nbors.flat[::K + 1] += R
        weights = linalg.solve(cov_nbors, np.ones(K).T, sym_pos=True)

        weights = weights / weights.sum()
        Weights[i, nbors[i]] = weights

    return (Weights)


def Y_(Weights, d):
    n, p = Weights.shape
    I = np.eye(n)
    m = (I - Weights)
    M = m.T.dot(m)

    eigvals, eigvecs = eigh(M, eigvals=(1, d), overwrite_a=True)
    ind = np.argsort(np.abs(eigvals))

    return (eigvecs[:, ind])


def LLE_(X, K):
    reg = 0.001
    nbors = Knbor_Mat(X, K)
    print(nbors[0])
    Weights = get_weights(X, nbors, reg, K)

    Y = Y_(Weights, 2)
    return (Y)


def plotter(K):
    fig = plt.figure(figsize=(10, 8))
    Y = LLE_(X, K)
    s = Y[test]
    plt.scatter(Y[:, 0], Y[:, 1], c=color, cmap=plt.cm.spectral)
    plt.show()
    plt.scatter(s[:, 0], s[:, 1], c="black")
    plt.show()


iris_ = datasets.load_iris()
iris = iris_.data
n_points = 1000
X, color = datasets.samples_generator.make_s_curve(n_points, random_state=0)
test = [354, 520, 246, 134, 3, 983, 186, 436, 893, 921]
plotter(100)