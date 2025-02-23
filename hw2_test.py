from hw2 import *
from hw2_utils import *
import torch
import matplotlib.pyplot as plt

def test_svm_solver(svm_solver, svm_predictor, degree=2, C=1.0, LR=0.01, num_iters=1000):
    # Generate XOR dataset
    x_train, y_train = xor_data()
    
    # Define polynomial kernel
    kernel = poly(degree)  # Ensure poly is correctly defined in hw2_utils
    
    # Generate test data (could be a separate test set)
    x_test, _ = xor_data()  # Using XOR data for testing, or you can create another dataset
    
    # Solve for alpha using the svm_solver function
    alpha = svm_solver(x_train, y_train, lr=LR, kernel=kernel, c=C, num_iters=num_iters)
    # Make predictions on the test set
    predictions = svm_predictor(alpha, x_train, y_train, x_test, kernel=kernel)
    
    # Print predictions
    print(predictions)
    
    # Plot decision boundary (optional)
    #plt.scatter(x_train[:, 0], x_train[:, 1], c=y_train, cmap='coolwarm', edgecolors='k', s=100)
    #svm_contour(svm_predictor, x_train, y_train, xmin=-2, xmax=2, ymin=-2, ymax=2)  # Adjust as needed

# Example usage (assuming your svm_solver and svm_predictor are implemented correctly)
test_svm_solver(svm_solver, svm_predictor, degree=2, C=1.0, num_iters=10000)
