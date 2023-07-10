from scipy.linalg import svd
import scipy.sparse as sps
import numpy as np
import math
import time

#M_data = [[1, 0, 0], [4, 0, 6], [7, 8, 0]]
M_data = [[1, 1, 1], [4, 1, 6], [7, 8, 1]]

M = np.array(M_data)
M_md = sps.csr_matrix(M_data)
W = np.array([[9, 8], [6, 5], [3, 2]])
H = np.array([[9, 8, 7], [6, 5, 4]])

# Frobenius Error function for dense matrices

def frob_error(x, y):
    return np.linalg.norm(x - y)

# RMSE Error function for dense matrices

def rmse_error(x, y):
    return np.linalg.norm(x - y) / np.sqrt(y.size)

# Percentage of Explained Variance function

def pev(egnval):

    # Sum of eigenvalues
    total_egnval = np.sum(egnval)

    # Explained variance
    var_exp = [((i / total_egnval) * 100) for i in sorted(total_egnval, reverse=True)]

    # Cumulative explained variance
    cumsum_exp = np.cumsum(var_exp)

    return var_exp, cumsum_exp


# Misidentification Rate function

def mr(A, B):

    # 1 - recall = 1 - (TP / (TP + FN))
    mr = (1 - (np.sum([(x == y == 0) for x, y in zip(A, B)]))) / np.count_nonzero(A==0)

    return mr


# Elastic net gradient descent function

def elastic_gradient_descent(X, y, n_iter=1000, tau=0.01, r=0.5, lmbd1=0.1, lmbd2=0.1, verbose=1):

    # Initialization
    n_samples, n_features = np.shape(X)
    coef = np.zeros(n_features)
    cost = np.zeros((n_iter))
    X = np.array(X)
    y = np.array(y)
    bias = 0.0

    start_time = time.time()

    # Gradient descent
    for i in range(n_iter):

        # Cost calculation
        y_pred = np.dot(X, coef) + bias
        cost[i] = (1 / n_samples) * np.sum((y_pred - y) ** 2)) + (((r * lmbd1) / n_samples) * np.sum(coef)) + (((((1 - r / 2) * lmbd2) / n_samples)) * np.sum(coef ** 2))

        if i % 50 == 0 and verbose == 1:
            print(f'Iteration: {i} - Time: {time.time() - start_time:.2f}s - Cost: {cost[i]:.3f}')

        # Gradient calculation
        dw = (-2 / n_samples) * (np.dot(X.T, (y_pred - y))) + (r * lmbd1) + ((lmbd2 * (1 - r)) * coef)
        db = (-2 / n_samples) * (np.sum(y_pred - y))

        # Parameters updated
        coef -= tau * dw
        bias -= tau * db

    return coef, bias, cost


# Lasso gradient descent function

def lasso_gradient_descent(X, y, n_iter=1000, tau=0.01, lmbd=0.1, verbose=1):

    # Initialization
    n_samples, n_features = np.shape(X)
    coef = np.zeros(n_features)
    cost = np.zeros((n_iter))
    X = np.array(X)
    y = np.array(y)
    bias = 0.0

    start_time = time.time()

    # Gradient descent
    for i in range(n_iter):

        # Cost calculation
        y_pred = np.dot(X, coef) + bias
        cost[i] = (1 / n_samples) * np.sum((y_pred - y) ** 2)) + ((lmbd / n_samples)) * np.sum(coef))

        if i % 50 == 0 and verbose == 1:
            print(f'Iteration: {i} - Time: {time.time() - start_time:.2f}s - Cost: {cost[i]:.3f}')

        # Gradient calculation
        dw = (-2 / n_samples) * (np.dot(X.T, (y_pred - y))) + lmbd
        db = (-2 / n_samples) * (np.sum(y_pred - y))

        # Parameters updated
        coef -= tau * dw
        bias -= tau * db

    return coef, bias, cost


# Singular value decomposition function

def svd(A):

    # SVD calculation
    U, s, VT = svd(A)

    # sigma initialization
    sigma = np.zeros((A.shape[0], A.shape[1]))

    # populate sigma with n x n diagonal matrix
    sigma[:A.shape[1], :A.shape[1]] = np.diag(s)

    # Reconstruct matrix
    B = U.dot(sigma.dot(VT))

    return U, s, VT, B

# Sparse Principal Component Analysis

def spca(A, reg='elastic', n_iter=1000, tau=0.01, r=0.5, lmbd1=0.1, lmbd2=0.1, verbose=1):

    # Update B (PC loading) given fixed A (PC weights)
    if reg == 'elastic':
        coef, bias, cost = elastic_gradient_descent(X, y)
    else:
        coef, bias, cost = lasso_gradient_descent(X, y)

    # Update A (PC weights) given fixed B (PC loading)
    U, s, VT, B = svd(A)

    # Eigenvectors ridge normalization

    return









