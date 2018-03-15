from sklearn.datasets import load_iris
from sklearn.cross_validation import train_test_split
from sklearn.neighbors.nearest_centroid import NearestCentroid
from sklearn import metrics
import numpy as np
from sklearn.datasets import make_moons
import functools
import pandas as pd
import math


def euclidean_distance(a,b):
    ans=0.0
    for i in range(0,len(a)):
        ans=ans+(a[i]-b[i])*(a[i]-b[i]);
    return math.sqrt(ans)

def cmppp(a,b):
    return a[0]-b[0]

def test(s,dic,types):
    lst=[]
    for i,j in dic.items():
        lst.append([euclidean_distance(j[0],s),i])
    #print(lst)
    lst=sorted(lst,key=functools.cmp_to_key(cmppp))
    ty=lst[0][1]
    '''cmean=dic[ty][0]
    c=dic[ty][1]
    cm=[]
    for i in range(len(cmean)):
        cm.append((cmean[i]*c+s[i])/(c+1))
    c=c+1
    dic[ty]=[cm,c]'''
    return ty

def train(X_train,y_train,types):
    dic={}
    for i in range(types):
        x=[0.0 for i in range(len(X_train[0]))]
        c=0
        for j in range(len(X_train)):
            if y_train[j]==i:
                x+=X_train[j]
                c+=1
        x=[i/c for i in x]
        dic[i]=[x,c]
    #print(dic)
    return dic
def centroid(X_train,X_test,y_train,y_test,types):
    dic=train(X_train,y_train,types)
    pred = []
    for i in X_test:
        pred.append(test(i,dic,types))
    #print(pred)
    pred=np.array(pred)
    acc=0.0
    for i in range(len(pred)):
        if pred[i]==y_test[i]:
            acc+=1
    #print("Accuracy is = ",acc/(float(len(y_test))))
    return acc/(float(len(y_test)))
if __name__ == '__main__':
    '''iris = load_iris()
    print(iris.feature_names)
    print(iris.data.shape)
    print(iris.target_names)
    print(type(iris.data), type(iris.target))
    X = iris.data
    y = iris.target'''
    X, y = make_moons(n_samples=3000, noise=0.1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4,random_state=5)
    #print(X_test.shape)
    types=2
    #X_train,y_train=make_moons(n_samples=2000,noise=0.4)
    #X_test,y_test=make_moons(n_samples=1000,noise=0.4)
    print("Accuracy is = ",centroid(X_train,X_test,y_train,y_test,2),end=' ')
    knearn=NearestCentroid()
    knearn.fit(X_train,y_train)
    print(metrics.accuracy_score(y_test,knearn.predict(X_test)))
