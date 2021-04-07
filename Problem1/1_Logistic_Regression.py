import math
import numpy as np
import matplotlib.pyplot as plt


def read_file(file_name):
    with open(file_name, "r") as fileX:
        lines = fileX.readlines()
    data = []
    for line in lines:
        line = line.split()
        line = [float(i) for i in line]
        if len(line) == 2:
            line.insert(0, 1)
        data.append(line)
    return data


def sigmoid(value):
    return 1 / (1 + math.exp(-value))


# theta is n*1 and x is n*1
def h(theta, x):
    return sigmoid(np.dot(np.transpose(theta), x))


# x(m*3) y(m*1) theta(3*1) grad_j(3*1)
def grad(x, y, theta):
    n = len(x[0])
    m = len(x)
    grad_j = []
    for j in range(0, n):
        value = 0
        for i in range(0, m):
            value += (1 - h(theta, np.transpose(y[i][0] * x[i]))) * y[i][0] * x[i][j]
        value *= (-1 / m)
        grad_j.append(value)
    grad_j = np.array(grad_j)
    return grad_j.reshape(grad_j.shape + (1,))


# x(m*3) y(m*1) theta(3*1) hessian_j(3*3)
def hessian(x, y, theta):
    n = len(x[0])
    m = len(x)
    hessian_j = np.zeros((n, n))
    for j in range(0, n):
        for k in range(0, n):
            value = 0
            for i in range(0, m):
                hh = h(theta, np.transpose(y[i][0] * x[i]))
                value += hh * (1 - hh) * x[i][k] * x[i][j]
            value *= (1 / m)
            hessian_j[j][k] = value
    return hessian_j


def main():
    # x(m*3)
    x = read_file("data/logistic_x.txt")
    x = np.array(x)
    y = read_file("data/logistic_y.txt")
    y = np.array(y)
    theta = [[0]] * 3
    theta = np.array(theta)

    print(theta)
    for i in range(0, 10):
        grad_j = grad(x, y, theta)
        hessian_j = hessian(x, y, theta)
        hessian_j_inv = np.linalg.inv(hessian_j)
        theta = theta - np.dot(hessian_j_inv, grad_j)
        print(theta)

    for i in range(0, x.shape[0]):
        plt.scatter(x[i, 1], x[i, 2], color='green' if y[i, 0] == 1 else 'red')
    x_line = np.linspace(0, 8, 100)
    y_line = (-theta[0] - x_line * theta[1]) / theta[2]
    plt.plot(x_line, y_line, '-b')
    plt.grid()
    plt.show()

main()
