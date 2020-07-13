from pointcloud import localcovold, localcov, localssq
import numpy as np
from numpy.random import normal

d = 4

x = np.empty((1000, d))
x[:,0] = normal(0,1,1000)
x[:,1] = normal(0,1,1000)
x[:,2] = 0.5 * x[:,0] + normal(0,1,1000)
x[:,3] = -0.5 * x[:,1] + normal(0,1,1000)

target = np.zeros(d)

cov1 = localcovold(x, target, 1, 40, 1e-4)
cov2 = localcov(x, target, 1, 40, 1e-4)
cov3 = np.cov(x)
cov4 = localssq(x, target, 1)

print('localssq n')
print(cov4.n)
