import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.cross_validation import train_test_split
from sklearn.datasets import make_moons
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from  sklearn import preprocessing
import sklearn.svm as sv
import pandas as pd



df=pd.read_csv('pima.csv',header=None)
print(df.head())

print(df.shape)


X=df.iloc[:,0:-1]
y=df.iloc[:,df.shape[1]-1:df.shape[1]]
print(X[0:5])
print(y[0:5])



scaler = preprocessing.StandardScaler()
X = scaler.fit_transform(X)
print(X[0:5])


#X,y=make_moons(n_samples=100,noise=0.1,random_state=3)
#mlp = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(40), random_state=1)
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.4,random_state=4);
print(X_train[0:5]);
mlp=MLPClassifier(hidden_layer_sizes=5)
mlp.fit(X_train,y_train);
print(accuracy_score(y_test,mlp.predict(X_test)))
print(mlp)
print(dir(mlp))

log=LogisticRegression()
log.fit(X_train,y_train);
print(accuracy_score(y_test,log.predict(X_test)))



svm=sv.SVC()
svm.fit(X_train,y_train)
print(accuracy_score(y_test,svm.predict(X_test)))



iris=load_iris()
X=iris.data
y=iris.target
print(X.shape)
print(iris.target_names)


mlp.fit(X,y);





