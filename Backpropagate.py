import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.datasets import make_moons
from sklearn.neural_network import MLPClassifier
X,y=make_moons(n_samples=100,noise=0.1,random_state=5)
#mlp = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(40), random_state=1)
mlp=MLPClassifier()
print(mlp)
mlp.fit(X,y)
print(y)
print(accuracy_score(y,mlp.predict(X)))
print(mlp)
nx=[i[0] for i in X]
ny=[i[1] for i in X]
colormap = np.array(['r', 'k'])
plt.scatter(nx, ny, c=colormap[y],norm=True,s=40)
#plt.show()


h = .02  # step size in the mesh
# create a mesh to plot in
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))


'''print(xx)
print(yy)
# Plot the decision boundary. For that, we will assign a color to each
# point in the mesh [x_min, m_max]x[y_min, y_max].
print(np.c_[xx.ravel(), yy.ravel()])
Z = mlp.predict(np.c_[xx.ravel(), yy.ravel()])
print(Z)
# Put the result into a color plot
Z = Z.reshape(xx.shape)
plt.contour(xx, yy, Z, cmap=plt.cm.Paired)
plt.show()'''

x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
h = 0.01
# Generate a grid of points with distance h between them
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
# Predict the function value for the whole gid
Z = mlp.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
print(Z)
# Plot the contour and training examples
plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Spectral)
plt.show()