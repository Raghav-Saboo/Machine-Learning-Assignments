from sklearn.datasets import make_moons
from sklearn.linear_model import Perceptron
from sklearn.datasets import load_iris
from sklearn.cross_validation import  train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import numpy as np
X,y=make_moons(n_samples=50,noise=0.3,random_state=4)
print(X,y)
print(type(X),type(y))
nx=[i[0] for i in X]
ny=[i[1] for i in X]
print(X)
colormap = np.array(['r', 'k'])
plt.scatter(nx, ny, c=colormap[y],norm=True,s=40)
per=Perceptron()
per.fit(X,y)
print(accuracy_score(per.predict(X),y))
x=np.linspace(min(nx),max(nx))
w1,w2=per.coef_[0]
c=per.intercept_[0]
y=[(-w1*i-c)/w2 for i in x]
plt.plot(x,y,'k-')
plt.show()
print(w1,w2,c)
a=[1,2,3]
print(a[:-1])