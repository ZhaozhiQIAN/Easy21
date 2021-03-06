import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import numpy as np


mc_V = np.loadtxt('mc_V.csv', delimiter=',')
fig = plt.figure()
ax = Axes3D(fig)
X = np.arange(0, 21)
Y = np.arange(0, 10)
X, Y = np.meshgrid(X, Y)
surf = ax.plot_surface(X, Y, mc_V, rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0, antialiased=False)
plt.xlabel('PlayerSum')
plt.ylabel('DealerCard')
plt.title('Value function for Monte-Carlo control')
plt.show()

plt.figure()
lambda0_mse = np.loadtxt('mse_trace0.csv', delimiter=',')
plt.plot(lambda0_mse)
plt.title('TD-0 MSE')
plt.xlabel('Iteration')
plt.ylabel('MSE')
plt.show()

plt.figure()
lambda1_mse = np.loadtxt('mse_trace1.csv', delimiter=',')
plt.plot(lambda1_mse)
plt.title('TD-1 MSE')
plt.xlabel('Iteration')
plt.ylabel('MSE')
plt.show()

plt.figure()
lambda0_mse_approx = np.loadtxt('approx_td_mse_trace0.csv', delimiter=',')
plt.plot(lambda0_mse_approx)
plt.title('Approximate TD-0 MSE')
plt.xlabel('Iteration')
plt.ylabel('MSE')
plt.show()

plt.figure()
lambda1_mse_approx = np.loadtxt('approx_td_mse_trace1.csv', delimiter=',')
plt.plot(lambda1_mse_approx)
plt.title('Approximate TD-1 MSE')
plt.xlabel('Iteration')
plt.ylabel('MSE')
plt.show()

plt.figure()
lambda_vs_mse = np.loadtxt('lambda_vs_mse.csv', delimiter=',')
plt.plot(np.arange(0, 1.1, 0.1), lambda_vs_mse)
plt.title('TD: lambda vs MSE')
plt.xlabel('lambda')
plt.ylabel('MSE')
plt.show()

plt.figure()
lambda_vs_mse_approx = np.loadtxt('lambda_vs_mse_approx.csv', delimiter=',')
plt.plot(np.arange(0, 1.1, 0.1), lambda_vs_mse_approx)
plt.title('Approximate TD: lambda vs MSE')
plt.xlabel('lambda')
plt.ylabel('MSE')
plt.show()

Q_learning_Q = np.load('Q_Learning_Q.npy')
Q_V = np.max(Q_learning_Q, axis=2)
fig2 = plt.figure()
ax = Axes3D(fig2)
X = np.arange(0, 21)
Y = np.arange(0, 10)
X, Y = np.meshgrid(X, Y)
surf = ax.plot_surface(X, Y, Q_V, rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0, antialiased=False)
plt.xlabel('PlayerSum')
plt.ylabel('DealerCard')
plt.title('Value function for Q Learning')
plt.show()

Dyna_Q_Q = np.load('Dyna_Q_Q.npy')
Q_V = np.max(Dyna_Q_Q, axis=2)
fig3 = plt.figure()
ax = Axes3D(fig3)
X = np.arange(0, 21)
Y = np.arange(0, 10)
X, Y = np.meshgrid(X, Y)
surf = ax.plot_surface(X, Y, Q_V, rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0, antialiased=False)
plt.xlabel('PlayerSum')
plt.ylabel('DealerCard')
plt.title('Value function for Dyna Q')
plt.show()

transition_mat = np.load('transition_mat.npy')

fig4 = plt.figure()
ax = Axes3D(fig4)
X = np.arange(0, 21)
Y = np.arange(0, 21)
X, Y = np.meshgrid(X, Y)
surf = ax.plot_wireframe(X, Y, transition_mat, rstride=1, cstride=1, linewidth=1, cmap=cm.coolwarm, antialiased=False)
plt.xlabel('From PlayerSum')
plt.xlabel('To PlayerSum')

