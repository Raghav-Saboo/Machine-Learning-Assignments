import pandas as pd
import numpy as np
from numpy import genfromtxt
from sklearn.datasets import make_moons
df=pd.read_csv('pima.csv',header=None)
my_data = genfromtxt('pima.csv', delimiter=',')
df=df.values
print(df.shape)
df=df[:,0]
print(df,df.shape,my_data[0].shape,type(df))
df = pd.read_csv('pima.csv', header=None)
df = df.values
col = df.shape[1]
X = df[:, 0:col - 1]
y = df[:, col - 1]
y = [int(i) for i in y]
norm_x = []
for i in X:
    mn = min(i)
    mx = max(i)
    na = [(j - mn) / (mx - mn) for j in i]
    norm_x.append(na)
norm_x = np.array(norm_x)
X = norm_x
print(X[0])
x,y=make_moons(n_samples=10,noise=0.5)
print(x)