import functools
import numpy as np
def cmp(a,b):
    return b[0]-a[0]

lis=[[1,2,5],[5,3,8],[9,4,6]]
lis.sort(key=functools.cmp_to_key(cmp))
print(lis)
lis=np.array(lis)

print(lis.dot(lis))
print(np.dot(lis,lis))

print(lis[:,[0,2]])
