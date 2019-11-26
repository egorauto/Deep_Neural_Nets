#%%
import csv
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
import time
import os
"""
logistic regression, assignment sheet 5

Authors: Georgy Antonov & Andrei Ilic

complete the code sections marked with TODO
"""

#%%
def tic():
    return time.time()

def toc(a):
    return time.time() - a

def load_data(filename):
    """load the data from the given file, returning a matrix for X and a vector for y"""
    xy = np.loadtxt(filename, delimiter=',')
    x = xy[:, 0:2]
    y = xy[:, 2]
    return x, y

# (a)
def plot_data(inputs, targets, ax=None, cols=('blue', 'red')):
    """ plots the data to a (possibly new) ax """
    ex1 = inputs[:, 1]
    ex2 = inputs[:, 2]

    if ax is None:
        # set up a new plot ax if we don't have one yet, otherwise we can plot to the existing one
        ax = plt.axes()
        # ax.set_xlim([-(np.amin(abs(ex1))+1), (np.amax(abs(ex1))+1)])
        # ax.set_ylim([-(np.amin(abs(ex2))+1), (np.amax(abs(ex2))+1)])

        ax.set_xlabel('Exam 1 score')
        ax.set_ylabel('Exam 2 score')
        plt.title('Student Admissions by Exam Scores')

    idcs = targets == 1

    ax.scatter(ex1[idcs], ex2[idcs], marker='x', c=cols[0], alpha=0.5, label='Admitted')
    ax.scatter(ex1[~idcs], ex2[~idcs], marker='o', c=cols[1], alpha=0.5, label='Not admitted')
    ax.legend(loc=1)
    return ax

# (b)
def sigmoid(x):
    return 1/(1+np.exp(-x))

# (c)
def cost(theta, inputs, targets, epsilon=1e-10):
    """ compute the cost function from the parameters theta """
    c = 0
    for i in range(inputs.shape[0]):
        c += targets[i]*np.log(sigmoid(theta @ inputs[i]) + epsilon) + (1-targets[i])*np.log(1-sigmoid(theta @ inputs[i]) + epsilon)
    return -c

# (c)
def gradient(theta, inputs, targets):
    """ compute the derivative of the cost function with respect to theta """
    g = np.zeros_like(theta)
    for j in range(theta.shape[0]):
        for i in range(inputs.shape[0]):
            g[j] += (sigmoid(theta @ inputs[i, :]) - targets[i])*inputs[i, j]
    return g

# (d)
def gradient_descent(theta_0, lr, steps, inputs, targets):
    """
    Args:
      theta_0: initial value for the parameters theta
      lr: learing rate
      steps: total number of iterations to perform
      inputs: training inputs
      targets: training targets
    returns the final value for theta
    """
    theta = theta_0.copy()
    for i in range(steps):
        g = gradient(theta, inputs, targets)
        c = cost(theta, inputs, targets)
        theta -= lr*g
        if i%10000 == 0:
            print('Iteration %d, cost: %.4f'%(i, c))
    return theta

# (e), (f)
def accuracy(inputs, targets, theta):
    pred = sigmoid(np.inner(theta, inputs))
    y_hat = pred >= 0.5
    return np.count_nonzero(targets == y_hat)/inputs.shape[0]

# (e), (f)
def add_boundary(inputs, ax, theta_trained, degree, num):
    w = theta_trained
    ex1 = inputs[:, 1]
    ex2 = inputs[:, 2]
    x_min = np.min(ex1)
    x_max = np.max(ex1)
    y_min = np.min(ex2)
    y_max = np.max(ex2)
    xx1 = np.linspace(x_min, x_max, num)
    xx2 = np.linspace(y_min, y_max, num)
    z = np.zeros((num, num))
    for i_x1, x1 in enumerate(xx1):
        for i_x2, x2 in enumerate(xx2):
            poly = polynomial_extension(np.array([x1, x2]).reshape(1, -1), degree)
            z[i_x2, i_x1] = sigmoid(np.inner(w, poly))
    xx1, xx2 = np.meshgrid(xx1, xx2)
    ax.contour(xx1, xx2, z, levels=[0.5])

# (f)
def polynomial_extension(inputs, degree):
    poly = PolynomialFeatures(degree = degree, interaction_only=False, include_bias=True)
    inputs_poly = poly.fit_transform(inputs)
    return inputs_poly

#%%
def main():

    if not os.path.exists('Figures'):
        os.mkdir('Figures')
        
    polynomial_degree = 3
    save_fig = True
    
    # load training and test sets
    train_inputs, train_targets = load_data('data_train.csv')
    test_inputs, test_targets = load_data("data_test.csv")

    # extend the input data in order to add a bias term to the dot product with theta
    train_inputs = np.column_stack([np.ones(len(train_targets)), train_inputs])
    test_inputs = np.column_stack([np.ones(len(test_targets)), test_inputs])

    # print('-'*100, '\ninputs\n', train_inputs)
    # print('-'*100, '\ntargets\n', train_targets)

    # (a) visualization
    ax = plot_data(train_inputs, train_targets, cols=('blue', 'red'), ax=None)
    # plt.savefig('Figures/Train_data.png')

    train_inputs_pol = polynomial_extension(train_inputs[:, 1:], degree=polynomial_degree)
    # print('-'*100, '\ninputs (polynomial extension)\n', train_inputs, '\n', '-'*100)

    # (d) use these parameters for training the model
    t = tic()
    theta_trained = gradient_descent(theta_0=np.zeros(len(train_inputs_pol[0, :])),
                                    lr=1e-4,
                                    steps=100000,
                                    inputs=train_inputs_pol,
                                    targets=train_targets)
    e_time = toc(t)
    print('Training done in %.2fs'%e_time)

    # (e) evaluation
    ax = plot_data(test_inputs, test_targets, cols=('lightblue', 'orange'), ax=ax)
    # plt.savefig('Figures/Test_data.png')
    test_inputs_pol = polynomial_extension(test_inputs[:, 1:], degree=polynomial_degree)
    acc = accuracy(test_inputs_pol, test_targets, theta_trained)
    print("Accuracy: %.3f"%acc)

    # (f) boundary plot
    add_boundary(train_inputs, ax=ax, theta_trained=theta_trained, degree=polynomial_degree, num=100)
    if save_fig:
        plt.savefig('Figures/Decision%d_acc%.3f.png'%(polynomial_degree, acc))
    plt.show()

#%%
if __name__ == '__main__':
    main()

