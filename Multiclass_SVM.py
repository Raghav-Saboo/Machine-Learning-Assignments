import numpy as np
import pandas as pd
from sklearn.datasets import make_moons
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
import cvxopt as cvx
from sklearn.datasets import load_iris
from sklearn.svm import SVC
import math



sigma=2.0;

def ker(a,b,kernel):
    ans=0;
    if kernel=='Linear':
        for i in range(len(b)):
            ans=ans+a[i]*b[i];
    elif kernel=='Gaussian':
        for i in range(len(b)):
            ans=ans+(a[i]-b[i])*(a[i]-b[i])
        ans=math.exp(-1.0*ans/(2*sigma*sigma))
    elif kernel=='Sigmoid':
        for i in range(len(b)):
            ans=ans+a[i]*b[i];
        ans=math.tanh(ans);
    return ans



def create_svm(X_train,y_train,type,eps,C):

    K = []
    for i in X_train:
        nl = []
        for j in X_train:
            nl.append(ker(i,j,type))
        K.append(nl)

    '''x = np.array(X_train)

    K = np.array([])

    if kernel == 'Linear':
        K = (1. + 1. / 1.0 * np.dot(x, x.T)) ** 1.0

    elif kernel == 'Gaussian':
        K = (1. + 1. / 1.0 * np.dot(x, x.T)) ** 1.0
        K=np.dot(x,x.T)**1.0
        N = K.shape[0]
        xsquared = (np.diag(K) * np.ones((1, N))).T
        b = np.ones((N, 1))
        K = K - 0.5 * ((np.dot(xsquared, b.T) + np.dot(b, xsquared.T)))
        K = np.exp(K / (2. * sigma ** 2))

    elif kernel == 'Sigmoid':
        K = (1. + 1. / 1.0 * np.dot(x, x.T)) ** 1.0
        K = np.tanh(K / len(X_train[0]) + 1.0)'''


    K = 1.0*np.array(K)

    #K = np.dot(np.array(X_train), np.array(X_train).transpose())

    P = np.dot(npytrain.transpose(),npytrain) * K

    P = 0.5 * P;

    #print(K.shape, npytrain.shape, (npytrain.transpose()).shape, P.shape)

    q = -np.ones((len(y_train), 1))

    A = 1.0*npytrain.reshape(1, len(y_train))
    #A=[npytrain.reshape(1, len(y_train)) for i in range(len(y_train))]


    #print('Size of A', A.shape)

    #b=np.zeros((len(y_train),1))
    b = 0.0;
    G1 = -np.eye(len(y_train))
    G2 = np.eye(len(y_train))

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
            b = b - supp_vec[j][0] * y_train[i] * ker(X_train[i], X_train[supp_vec[j][1]],type)

    b = b / len(supp_vec)

    return [supp_vec,b]







iris=load_iris()
X=iris.data;
y=iris.target;
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.4,random_state=4)
#X,y=make_moons(n_samples=500,noise=0.1,random_state=2)
#print(y)

no_of_class=np.unique(y)

y0=[]
y1=[]
y2=[]
#y_train=y;
for i in range(len(X_train)):
    if y_train[i]==0:
        y0.append(1)
        y1.append(-1)
        y2.append(-1)
    elif y_train[i]==1:
        y1.append(1)
        y2.append(-1)
        y0.append(-1)
    else:
        y2.append(1)
        y0.append(-1)
        y1.append(-1)

#X_train=X;

'''y1=y;
for i in range(len(y)):
    if y[i]==0:
        y1[i]=-1
'''

kernel='Gaussian'

npytrain=np.array(y0)
npytrain=npytrain.reshape(1,len(y0))
svm_own0=create_svm(X_train,y0,kernel,1e-2,1.0)



npytrain=np.array(y1)
npytrain=npytrain.reshape(1,len(y1))
svm_own1=create_svm(X_train,y1,kernel,1e-2,1.0)


npytrain=np.array(y2)
npytrain=npytrain.reshape(1,len(y2))
svm_own2=create_svm(X_train,y2,kernel,1e-2,1.0)



#X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.4,random_state=2)

op=[]
for j in range(len(y_test)):
    pr=0.0;
    out=[]
    supp_vec = svm_own0[0]
    b = svm_own0[1];
    for i in range(len(supp_vec)):
        pr=pr+y0[supp_vec[i][1]]*supp_vec[i][0]*ker(X_test[j],X_train[supp_vec[i][1]],kernel);
    pr+=b;
    out.append(pr)

    #print(pr, end=' ')

    pr=0.0;
    supp_vec = svm_own1[0]
    b = svm_own1[1];
    for i in range(len(supp_vec)):
        pr = pr + y1[supp_vec[i][1]] * supp_vec[i][0] * ker(X_test[j], X_train[supp_vec[i][1]],kernel);
    pr += b;
    out.append(pr)

    #print(pr,end=' ')

    pr=0.0;
    supp_vec = svm_own2[0]
    b = svm_own2[1];
    for i in range(len(supp_vec)):
        pr = pr + y2[supp_vec[i][1]] * supp_vec[i][0] * ker(X_test[j], X_train[supp_vec[i][1]],kernel);
    pr += b;

    #print(pr)

    out.append(pr)
    out=np.array(out)
    op.append(np.argmax(out))
    #print(out,np.argmax(out))

#op=np.array(op)
#op=np.sign(op)



#print(op)
print('My SVM score = ',accuracy_score(y_test,op))

svm=SVC(kernel='rbf');
#print(svm)
svm.fit(X_train,y_train);
print('Built in svm score = ',accuracy_score(y_test,svm.predict(X_test)))



