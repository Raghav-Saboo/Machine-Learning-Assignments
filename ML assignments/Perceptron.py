import pandas as pd
from sklearn.datasets import make_moons
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import numpy as np
def get_prediction(w,x):
    #print(w,x)
    ans=w[0]
    for i in range(len(x)):
        ans+=w[i+1]*x[i]
    if ans >= 0.0:
        return 1
    else:
        return 0

def train_network(X,y,learning_rate,n_epochs):
    w = [0.0 for i in range(len(X[0]) + 1)]
    for i in range(n_epochs):
        for j in range(len(X)):
            pred=get_prediction(w,X[j])
            actual=y[j]
            w[0]=w[0]+learning_rate*(actual-pred)
            for k in range(len(w)-1):
                w[k+1]=w[k+1]+learning_rate*(actual-pred)*X[j][k]
    return w

def train_network_batch(X,y,learning_rate,n_epochs):
    w = [0.0 for i in range(len(X[0]) + 1)]
    for i in range(n_epochs):
        pred_arr=[]
        for j in range(len(X)):
            pred=get_prediction(w,X[j])
            actual=y[j]
            pred_arr.append(pred)
        for j in range(len(pred_arr)):
            w[0]=w[0]+learning_rate*(y[j]-pred_arr[j])
        for k in range(len(w) - 1):
            for j in range(len(pred_arr)):
                w[k+1]=w[k+1]+learning_rate*(y[j]-pred_arr[j])*X[j][k]
    return w

def perceptron(X,y,learning_rate,n_epochs):
    w=train_network(X,y,learning_rate,n_epochs)
    acc=0.0
    for i in range(len(y)):
        if get_prediction(w,X[i])==y[i]:
            acc+=1
    return [w,acc/len(y)]

def batch_perceptron(X,y,learning_rate,n_epochs):
    w=train_network_batch(X,y,learning_rate,n_epochs)
    acc=0.0
    for i in range(len(y)):
        if get_prediction(w,X[i])==y[i]:
            acc+=1
    return [w,acc/len(y)]

def train_network_mini_batch(X,y,learning_rate,n_epochs,w):
    for i in range(n_epochs):
        pred_arr=[]
        for j in range(len(X)):
            pred=get_prediction(w,X[j])
            actual=y[j]
            pred_arr.append(pred)
        for j in range(len(pred_arr)):
            w[0]=w[0]+learning_rate*(y[j]-pred_arr[j])
        for k in range(len(w) - 1):
            for j in range(len(pred_arr)):
                w[k+1]=w[k+1]+learning_rate*(y[j]-pred_arr[j])*X[j][k]
    return w

def mini_batch_perceptron(X,y,learning_rate,n_epochs,k):
    w = [0.0 for i in range(len(X[0]) + 1)]
    j=0
    for i in range(k):
        if j<len(X):
            w=train_network_mini_batch(X[j:max(len(X),j+k)],y[j:max(len(y),j+k)],learning_rate,n_epochs,w)
        j+=k
    acc=0.0
    for i in range(len(y)):
        if get_prediction(w,X[i])==y[i]:
            acc+=1
    #print(len(y)-acc)
    return [w,acc/len(y)]

if __name__ == '__main__':
    '''df=pd.read_csv('sonar.all-data.csv',header=None)
    col=df.shape[1]
    df,y=df.loc[:,0:col-2],df.loc[:,col-1]
    def fun(x):
        if x=='R':
            return 1
        else:
            return 0
    y=[fun(x) for x in y]
    y=np.array(y)
    #print(y,type(y))
    X=np.array(df)'''
    #fun = mini_batch_perceptron(X, y, 0.01,100,5)
    #print('Accuracy of perceptron is = ' + str(100 * fun[1]) + "%")
    X, y = make_moons(n_samples=50, noise=0.3, random_state=4)
    fun=perceptron(X,y,0.01,10)
    batch=batch_perceptron(X,y,0.01,20)
    mini_batch=mini_batch_perceptron(X,y,0.01,20,5)
    print('Accuracy of SGD is = '+str(fun[1]*100)+"%")
    print('Accuracy of Batch perceptron is = '+str(100*batch[1])+"%")
    print('Accuracy of Mini Batch Perceptron is = '+str(100*mini_batch[1])+"%")
    per=Perceptron()
    per.fit(X,y)
    print('Accuracy of SK learn is = ' + (str(100*accuracy_score(y,per.predict(X))))+"%")
    nx=[i[0] for i in X]
    ny=[i[1] for i in X]
    colormap = np.array(['r', 'k'])
    plt.scatter(nx, ny, c=colormap[y],norm=True,s=40)
    x = np.linspace(min(nx), max(nx))

    w1, w2 = per.coef_[0]
    c = per.intercept_[0]
    y = [(-w1 * i - c) / w2 for i in x]
    plt.plot(x, y, 'k-',label='Sklearn')

    w1, w2 = fun[0][1],fun[0][2]
    c = fun[0][0]
    y = [(-w1 * i - c) / w2 for i in x]
    plt.plot(x, y, 'r-',label='SGD')

    w1, w2 = batch[0][1],batch[0][2]
    c = batch[0][0]
    y = [(-w1 * i - c) / w2 for i in x]
    plt.plot(x, y, 'g-',label='Batch')

    w1, w2 = mini_batch[0][1],mini_batch[0][2]
    c = mini_batch[0][0]
    y = [(-w1 * i - c) / w2 for i in x]
    plt.plot(x, y, 'b-',label='Mini Batch')

    plt.legend()
    plt.show()
