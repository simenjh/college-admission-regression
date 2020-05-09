import numpy as np
import matplotlib.pyplot as plt



def plot_X(X, y):
    X_reduce = PCA(X)
    plt.scatter(X_reduce, y)
    plt.xlabel("Reduced features")
    plt.ylabel("Chance of admission")
    plt.show()


def PCA(X):
    [U, s, v] = np.linalg.svd(X.T)
    U_reduce = U[:, 0:1]
    X_reduce = np.dot(U_reduce.T, X.T)
    return X_reduce
    

def plot_cost(J_vals):
    J_length = len(J_vals)
    plt.plot(list(range(J_length)), J_vals)
    plt.xlabel("No. of iterations")
    plt.ylabel("Cost, J")
    plt.show()
