import numpy as np
import matplotlib.pyplot as plt
xx, yy = np.meshgrid(np.arange(0, 5, 0.5), np.arange(0, 5, 0.5))
print(xx)
print(yy)

x = [1,2,3,4]
y = [3,4,8,6]

#matplotlib.pyplot.
plt.scatter(x,y)

#plt.plot(x,y)
plt.show()