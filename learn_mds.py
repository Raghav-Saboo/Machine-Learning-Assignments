from sklearn.manifold import mds
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

dis=[[0,206,429,1504,963,2976,3095,2979,1949],[206,0,233,1308,802,2815,2934,2786,1771],[429,233,0,1075,671,2684,2799,2631,1616],[1504,1308,1075,0,1329,3273,3053,2687,2037],[963,802,671,1329,0,2013,2142,2054,996],[2976,2815,2684,3273,2013,0,808,1131,1307],[3095,2934,2799,3053,2142,808,0,379,1235],[2979,2786,2631,2687,2054,1131,379,0,1059],[1949,1771,1616,2037,996,1307,1235,1059,0]]


for i in dis:
    print(i)

for i in range(len(dis)):
    for j in range(len(dis)):
        if dis[i][j]!=dis[j][i]:
            print(i,j,dis[i][j])

dr=mds.MDS(n_components=2,dissimilarity="precomputed")
print(dr.fit_transform(dis))


y=dr.fit_transform(dis)

nx=[]
for i in y:
    nx.append([i[0],i[1]])

nx=np.array(nx)

#nx=StandardScaler(nx)
scaler=StandardScaler()
nx=scaler.fit_transform(nx)
print()
print(nx)
nx=np.array(nx)


plt.plot([i[0] for i in nx],[i[1] for i in nx],'o')
plt.show()