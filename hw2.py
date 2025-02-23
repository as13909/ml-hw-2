import hw2_utils as utils
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

def svm_solver(x_train, y_train, lr, num_iters,
               kernel=utils.poly(degree=1), c=None):
    '''
    Computes an SVM given a training set, training labels, the number of
    iterations to perform projected gradient descent, a kernel, and a trade-off
    parameter for soft-margin SVM.

    Arguments:
        x_train: 2D tensor with shape (n, d).
        y_train: 1D tensor with shape (n,), whose elements are +1 or -1.
        lr: Learning rate.
        num_iters: Number of gradient descent steps.
        kernel: Kernel function. Default: polynomial of degree 2.
        c: Trade-off parameter for soft-margin SVM. Default: None (hard-margin).

    Returns:
        alpha: 1D tensor with shape (n,), optimal dual solution (detached).
    '''
    n = x_train.shape[0]

    # Initialize alpha with requires_grad=True
    alpha = torch.zeros(n, requires_grad=True)

    # Compute Kernel Matrix using vectorized operations
    K = torch.zeros((n, n), dtype=torch.float)
    for i in range(n):
        for j in range(n):
            K[i, j] = kernel(x_train[i].unsqueeze(0), x_train[j].unsqueeze(0))  # Ensure correct tensor shape


    for _ in range(num_iters):
        # Compute the gradient: ∇f(α) = K (α ∘ y) ∘ y - 1
        grad = torch.matmul(K, alpha * y_train) * y_train - 1
        # Use torch.no_grad() when updating alpha to avoid tracking gradients
        with torch.no_grad():
            alpha -= lr * grad
            alpha.clamp_(min=0, max=c if c is not None else float('inf'))

        # Ensure alpha retains gradient tracking
        alpha.requires_grad_()
        
        if _ % 100 == 0 or i == num_iters - 1:
            print(f"Iteration {_}: Max α = {alpha.max().item()}, Min α = {alpha.min().item()}")

        # Debugging output)
    return alpha.detach()




def svm_predictor(alpha, x_train, y_train, x_test, kernel):
    '''
    Returns the kernel SVM's predictions for x_test using the SVM trained on
    x_train, y_train with computed dual variables alpha.

    Arguments:
        alpha: 1d tensor with shape (n,), denoting an optimal dual solution.
        x_train: 2d tensor with shape (n, d), denoting the training set.
        y_train: 1d tensor with shape (n,), whose elements are +1 or -1.
        x_test: 2d tensor with shape (m, d), denoting the test set.
        kernel: The kernel function.

    Return:
        A 1d tensor with shape (m,), the outputs of SVM on the test set.
    '''
    # Compute the kernel matrix for the test set against the training set
    train_mean = x_train.mean(dim=0)
    train_std = x_train.std(dim=0)
    x_train = (x_train - train_mean) / train_std
    x_test = (x_test - train_mean) / train_std  # Ensure test set uses training mean/std

    K_test = torch.empty(x_test.shape[0], x_train.shape[0])

    for i in range(x_test.shape[0]):
        for j in range(x_train.shape[0]):
            K_test[i, j] = utils.poly_implementation(x_test[i], x_train[j],2)

    # Compute the SVM decision scores
    scores = torch.matmul((alpha * y_train), K_test.T)

    return scores.detach()


