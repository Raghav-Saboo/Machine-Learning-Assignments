import numpy as np
import pandas as pd
from sklearn.datasets import make_moons
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
import cvxopt as cvx
from sklearn.svm import SVC
import math


sigma=0.25;

def ker(a,b,type):
    ans=0;
    if type=='Linear':
        for i in range(len(b)):
            ans=ans+a[i]*b[i];
    elif type=='Gaussian':
        for i in range(len(b)):
            ans=ans+(a[i]-b[i])*(a[i]-b[i])
        ans=math.exp(-1.0*ans/(2*sigma*sigma))
    elif type=='Sigmoid':
        for i in range(len(b)):
            ans=ans+a[i]*b[i];
        ans=math.tanh(ans)
    return ans


def predict(supp_vec,b,X_train,X_test,kernel):
    op=[]
    for j in range(len(y_test)):
        pr = 0.0;
        for i in range(len(supp_vec)):
            pr = pr + y_train[supp_vec[i][1]] * supp_vec[i][0] * ker(X_test[j],X_train[supp_vec[i][1]],kernel);#K[j][supp_vec[i][1]];
        pr += b;
        op.append(pr)

    op = np.array(op)
    op = np.sign(op)
    return op


def predict_1(svm,kernel,X1):
    supp_vec=svm[0]
    b=svm[1]
    op=[]
    for j in range(len(X1)):
        pr = 0.0;
        for i in range(len(supp_vec)):
            pr = pr + y_train[supp_vec[i][1]] * supp_vec[i][0] * ker(X1[j], X_train[supp_vec[i][1]],kernel);
        pr += b;
        op.append(pr)
    return np.sign(np.array(op))


def create_svm(X_train,y_train,kernel,eps,C):

    '''K = []
    for i in X_train:
        nl = []
        for j in X_train:
            nl.append(ker(i, j,kernel))
        K.append(nl)

    K = np.array(K)'''


    x=np.array(X_train)

    K=np.array([])

    if kernel=='Linear':
        K=(1.+1./1.0*np.dot(x,x.T))**1.0

    elif kernel=='Gaussian':
        K = (1. + 1. / 1.0 * np.dot(x, x.T)) ** 1.0
        N=K.shape[0]
        xsquared = (np.diag(K) * np.ones((1,N))).T
        b = np.ones((N, 1))
        K = K - 0.5*( (np.dot(xsquared, b.T) + np.dot(b,xsquared.T)))
        K = np.exp(K / (2. * sigma ** 2))

    elif kernel=='Sigmoid':
        K=(1.+1./1.0*np.dot(x,x.T))**1.0
        K=np.tanh(K/len(X_train[0])+1.0)

    P = npytrain * npytrain.transpose() * K

    P = 0.5 * P;

    #print(K.shape, npytrain.shape, (npytrain.transpose()).shape, P.shape)

    q = -np.ones((len(y_train), 1))

    A = 1.0 * npytrain.reshape(1, len(y_train))

    #print('Size of A', A.shape)

    # b=np.zeros((len(y_train),1))
    b = 0.0;
    G1 = -np.eye(len(y_train))
    G2 = np.eye(len(y_train))

    #C = 1.0
    G = np.concatenate((G2, G1));
    # G=G1;

    h1 = C * np.ones((len(y_train), 1));
    h2 = np.zeros((len(y_train), 1));

    h = np.concatenate((h1, h2));
    # h=h2;

    #print(G.shape, h.shape)

    # b=0.0

    lamb = cvx.solvers.qp(cvx.matrix(P), cvx.matrix(q), cvx.matrix(G), cvx.matrix(h), cvx.matrix(A), cvx.matrix(b))

    lamb = lamb['x']

    lamb = np.array(lamb)

    #print(lamb)

    b = 0.0

    #eps = 1e-5;

    #print(eps)

    supp_vec = []

    for i in range(len(y_train)):
        if lamb[i] > eps:
            supp_vec.append([lamb[i],i])

    #print(lamb.shape, len(supp_vec))

    # print(X_train[0])

    b = 0.0
    for j in range(len(supp_vec)):
        b=b+y_train[supp_vec[j][1]]
        for i in range(len(X_train)):
            b = b - supp_vec[j][0] * y_train[i] * ker(X_train[i], X_train[supp_vec[j][1]],kernel)

    b = b / len(supp_vec)

    return [supp_vec,b]








X,y=make_moons(n_samples=100,noise=0.1,random_state=5)


y1=y;
for i in range(len(y)):
    if y[i]==0:
        y1[i]=-1

X_train,X_test,y_train,y_test=train_test_split(X,y1,test_size=0.4,random_state=1)

npytrain=np.array(y_train)
npytrain=npytrain.reshape(1,len(y_train))

kernel='Sigmoid'
svm_own=create_svm(X_train,y_train,kernel,1e-5,0.01)

supp_vec=svm_own[0]

b=svm_own[1];

op=predict(supp_vec,b,X_train,X_test,kernel)
'''op=[]
for j in range(len(y_test)):
    pr=0.0;
    for i in range(len(supp_vec)):
        pr=pr+y_train[supp_vec[i][1]]*supp_vec[i][0]*ker(X_test[j],X_train[supp_vec[i][1]],kernel);
    pr+=b;
    op.append(pr)'''

op=np.array(op)
op=np.sign(op)


acc=accuracy_score(y_test,op)
print(acc)
acc*=100;

svm=SVC(kernel='sigmoid');
print(svm)
svm.fit(X_train,y_train);
print(accuracy_score(y_test,svm.predict(X_test)))

x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
h = 0.02
# Generate a grid of points with distance h between them
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
# Predict the function value for the whole gid
Z = (predict_1(svm_own,kernel,np.c_[xx.ravel(), yy.ravel()]))
Z = Z.reshape(xx.shape)
print(Z)
# Plot the contour and training examples
plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Spectral)
plt.title('Sigmoid Kernel with accuracy '+str(acc)+"%")
plt.xlabel('Feature 1 ---> ')
plt.ylabel('Feature 2 ---> ')
plt.show()