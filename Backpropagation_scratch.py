import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
import math
from sklearn.cross_validation import train_test_split
import time
from sklearn.neural_network import MLPClassifier
import random
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
import copy
from sklearn.datasets import load_iris
from sklearn.datasets import make_moons

def sum(weights,inputs):
    ans=weights[-1]
    for i in range(len(inputs)):
        ans+=weights[i]*inputs[i]
    return ans

def activation(val,activation_function):
    if activation_function=='relu':
        try:
            return math.log(1 + math.exp(val), math.e)
        except OverflowError:
            ans = float('inf')
            return ans
    elif activation_function=='sigmoid':
        return 1.0/(1.0+math.exp(-1*val))
    elif activation_function=='tanh':
        try:
            return ((math.exp(val) - math.exp(-1 * val) / (math.exp(val) + math.exp(-1 * val))))
        except OverflowError:
            ans = float('inf')
            return ans
    elif activation_function=='leaky_relu':
        if val>0:
            return val
        else:
            return 0.01*val
    else:
        if val>0:
            return val
        else:
            return math.exp(val)-1

def derivative(val,activation_function):
    if activation_function=='relu':
        if val<=0:
            return 0
        else:
            return 1
    elif activation_function=='sigmoid':
        return val*(1-val)
    elif activation_function=='tanh':
        return 1-(val*val)
    elif activation_function=='leaky_relu':
        if val>0:
            return 1
        else:
            return 0.01
    else:
        if val>0:
            return 1
        else:
            return val+1

def forward_propagate(network,data,activation_function):
    inp=data
    for layer in network:
        ninp=[]
        for neuron in layer:
            neuron['output']=activation(sum(neuron['weights'],inp),activation_function)
            ninp.append(neuron['output'])
        inp=copy.deepcopy(ninp)
    return inp

def backward_propagate(network,expected,activation_function):
    for i in range(len(network)-1,-1,-1):
        layer=network[i]
        if i!=len(network)-1:
            for j in range(len(layer)):
                err=0.0
                for neuron in network[i+1]:
                #    print(i,j,neuron)
                    err+=neuron['delta']*neuron['weights'][j]
                neuron=layer[j]
                neuron['delta']=err*derivative(neuron['output'],activation_function)
        else:
            for j in range(len(layer)):
                neuron=layer[j]
                neuron['delta']=(expected[j]-neuron['output'])*derivative(neuron['output'],activation_function)
               # print(i,j,neuron)
def update_weights(network,learning_rate,data):
    inp=data
    for i in range(len(network)):
        layer=network[i]
        ninp=[]
        for neuron in layer:
            for j in range(len(neuron['weights'])):
                if j!=len(neuron['weights'])-1:
                    neuron['weights'][j]+=learning_rate*neuron['delta']*inp[j]
                else:
                    neuron['weights'][j] += learning_rate * neuron['delta']
        ninp=[neuron['output'] for neuron in network[i]]
        inp=copy.deepcopy(ninp)

def MLP_classifier(n_inputs,n_outputs,n_hidden,n_epochs,learning_rate,activation_function,X,y):
    network=[]
    for i in range(len(n_hidden)):
        if i==0:
            hidden=[{'weights':[random.random() for j in range(n_inputs+1)]}for k in range(n_hidden[i])]
            network.append(hidden)
        else:
            hidden = [{'weights': [random.random() for j in range(n_hidden[i-1] + 1)]} for k in range(n_hidden[i])]
            network.append(hidden)
    output = [{'weights': [random.random() for j in range(n_hidden[len(n_hidden)-1] + 1)]} for k in range(n_outputs)]
    network.append(output)
    for layer in network:
        print(layer)

    for i in range(n_epochs):
        for j in range(len(X)):
            data=X[j]
            outputs=forward_propagate(network,data,activation_function)
            op=[0 for k in range(n_outputs)]
            op[y[j]]=1
            backward_propagate(network,op,activation_function)
            update_weights(network,learning_rate,data)
    return network

def predict(network,X,y,activation_function):
    acc=0.0
    j=0
    for i in X:
        op=forward_propagate(network,i,activation_function)
        #print(op.index(max(op)))
        print(i,y[j],op.index(max(op)),op)
        if op.index(max(op))==y[j]:
            acc+=1
            #print(op.index(max(op)))
        j=j+1
    return acc/len(y)

def predict_1(network,X,activation_function):
    ans=[]
    j=0
    for i in X:
        op=forward_propagate(network,i,activation_function)
        ans.append(op.index(max(op)))
        #print(i,op.index(max(op)),op)
    return np.array(ans)

if __name__ == '__main__':
    X,y=make_moons(n_samples=100,noise=0.1,random_state=4)
    #iris = load_iris()
    #X = iris.data
    #y = iris.target
    print(X[0:5])
    '''df = pd.read_csv('pima.csv', header=None)
    df = df.values
    col = df.shape[1]
    X = df[:, 0:col - 1]
    y = df[:, col - 1]
    y = [int(i) for i in y]
    norm_x=[]
    for i in X:
        mn=min(i)
        mx=max(i)
        na=[(j-mn)/(mx-mn) for j in i]
        norm_x.append(na)
    norm_x=np.array(norm_x)
    X=norm_x'''
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4,random_state=4)
    lx = [0.01, 0.02, 0.03, 0.04, 0.1, 0.2, 0.3, 0.4]
    ly=[]
    slx=lx
    sly=[]
    '''for le in lx:
        mlp=MLP_classifier(len(X_train[0]),len(set([i for i in y_train])),[12],200,le,'erlu',X_train,y_train)
        acc=predict(mlp,X_test,y_test,'erlu')
        ly.append(acc)
        print(acc,end=' ')
        skmlp=MLPClassifier(learning_rate_init=le,hidden_layer_sizes=(12,),activation='relu',solver='sgd')
        skmlp.fit(X_train,y_train)
        #print(skmlp)
        sly.append(accuracy_score(y_test,skmlp.predict(X_test)))
        print(sly[len(sly)-1])'''

    start_time = time.time()
    mlp = MLP_classifier(len(X_train[0]), len(set([i for i in y_train])), [16], 300, 0.001, 'erlu', X_train,y_train)
    print("--- %s seconds ---" % (time.time() - start_time), end=' ')
    acc = predict(mlp, X_test, y_test, 'erlu')
    print(acc, end=' ')
    skmlp = MLPClassifier()
    print(skmlp)
    skmlp.fit(X_train, y_train)
    # print(skmlp)
    print(accuracy_score(y_test, skmlp.predict(X_test)))



    '''plt.ylim([0,1])
    plt.plot(lx,ly,'g-',label='Our own')
    plt.plot(slx, sly,'b-',label='SK learn')
    plt.xlabel("Learning rate")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show()'''

    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    h = 0.02
    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # Predict the function value for the whole gid
    Z = (predict_1(mlp,np.c_[xx.ravel(), yy.ravel()],'erlu'))
    Z = Z.reshape(xx.shape)
    print(Z)
    # Plot the contour and training examples
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Spectral)
    plt.show()