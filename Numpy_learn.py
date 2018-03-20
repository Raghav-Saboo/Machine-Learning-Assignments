import numpy as np

a=np.array([[1,1,1],[2,2,2],[3,3,3]])
print(a.diagonal().shape)
print((a.diagonal()*np.ones((1,3))).shape)

y=np.array([[2,2],[3,3]])
print(np.sum(y ** 2, axis=1))