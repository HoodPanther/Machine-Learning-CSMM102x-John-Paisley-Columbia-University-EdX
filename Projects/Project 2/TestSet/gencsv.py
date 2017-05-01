# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.mlab import bivariate_normal



# K classes for y, pi for the discrete distribution of y
K = 6
pi = np.array([4., 8., 15., 16., 23., 42.])
pi /= pi.sum()



# d dimensions for x, mu and Sigma for the bivariate normal distribution of x|y
d = 2
mu = np.array([(0., 9.),
               (1., 2.),
               (8., 6.),
               (1., 4.),
               (1., 0.),
               (9., 1.)])
Sigma = [np.diag([i]*d) for i in [1., 1.5, 2., 2.5, 3., 3.5]]



# Generate CSVs for our training program
n = 1000
y_train = np.random.choice(range(K), size=n, p=pi)
x_train = np.zeros((n,d))
for y in range(K):
    index, = np.where(y_train == y)
    x_train[index,:] = np.random.multivariate_normal(mu[y,:], Sigma[y], len(index))
np.savetxt("X_train.csv", x_train, delimiter=",")
np.savetxt("y_train.csv", y_train, fmt="%d")

n_test = 5
x_test = np.random.uniform(0., 10., (n_test, d))
np.savetxt("X_test.csv", x_test, delimiter=",")



# Plot the expected result of the Bayes classifier
x = np.arange(-5., 15., .1)
y = np.arange(-5., 15., .1)
X, Y = np.meshgrid(x, y, indexing='xy')

Z_array = []
for i in range(K):
    mu_i = mu[i]
    Sigma_i = Sigma[i]    
    Z = bivariate_normal(X, Y,
                        sigmax = np.sqrt(Sigma_i[0,0]),
                        sigmay = np.sqrt(Sigma_i[1,1]),
                        mux = mu_i[0],
                        muy = mu_i[1],
                        sigmaxy = Sigma_i[0,1]) \
        * pi[i]
    Z_array.append(Z)
Z_array = np.array(Z_array)
imax = np.argmax(Z_array, axis=0)

plt.contourf(X, Y, imax, levels=range(K+1))
plt.colorbar()

for i in range(K):
    plt.annotate(str(i), xy=mu[i,:], xytext=(-20, -20),
        textcoords='offset points', ha='right', va='bottom',
        bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
        arrowprops=dict(arrowstyle = '->', connectionstyle='arc3,rad=0'))

plt.show()