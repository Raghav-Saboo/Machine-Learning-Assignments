from sklearn.datasets import load_iris
from sklearn.cross_validation import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
import numpy as np
import functools
import matplotlib.pyplot as pt
import pandas as pd
import math


def euclidean_distance(a,b):
    ans=0.0
    #print(a,b)
    for i in range(0,len(a)):
        ans=ans+(a[i]-b[i])*(a[i]-b[i]);
    return math.sqrt(ans)

def cmppp(a,b):
    return a[0]-b[0]

def test(s,X_train,y_train,k,types):
    lst=[]
    for i in range(0,len(X_train)):
        lst.append([euclidean_distance(X_train[i],s),y_train[i]])
    #print(lst)
    lst=sorted(lst,key=functools.cmp_to_key(cmppp))
    cnt=[0 for i in range(types)]
    for i in range(k):
        cnt[lst[i][1]]+=1
    #print(cnt)
    mx=0
    ty=0
    for i in range(0,types):
        if mx < cnt[i]:
            mx=cnt[i]
            ty=i
    return ty

def knn(X_train,X_test,y_train,y_test,k,types):
    pred = []
    for i in X_test:
        pred.append(test(i,X_train,y_train,k,types))
    #print(pred)
    pred=np.array(pred)
    acc=0.0
    for i in range(len(pred)):
        if pred[i]==y_test[i]:
            acc+=1
    #print("Accuracy is = ",acc/(float(len(y_test))))
    return acc/(float(len(y_test)))
if __name__ == '__main__':
    iris = load_iris()
    print(iris.feature_names)
    print(iris.data.shape)
    print(iris.target_names)
    print(type(iris.data), type(iris.target))
    X = iris.data
    y = iris.target
    ''' df=pd.read_csv('pima.csv',header=None)
    df=df.values
    col=df.shape[1]
    X=df[:,0:col-1]
    y=df[:,col-1]
    y=[int(i) for i in y]
    y=np.array(y)
    print(X.shape,y.shape)'''
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4,random_state=3)
    types=len(iris.target_names)
    pt.ylabel('Accuracy')
    pt.xlabel('K neighbors')
    krange=[]
    accuracy=[]
    for i in range(1,31):
        acc=knn(X_train,X_test,y_train,y_test,i,types)
        accuracy.append(acc)
        krange.append(i)
        print(i,acc,end=' ')
        knearn=KNeighborsClassifier(n_neighbors=i)
        knearn.fit(X_train,y_train)
        print(metrics.accuracy_score(y_test,knearn.predict(X_test)))
    pt.plot(krange,accuracy)
    pt.show()