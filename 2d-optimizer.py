import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import sys

def func(x, y):
    return np.sin(x)*np.cos(y)

def nearest(x, y, X, Y):
    return np.abs(x - X).argmin(), np.abs(y - Y).argmin()

def newton(x, y, Z, x0, y0, alpha, iterations):
    record = []
    delta_x = x[1] - x[0]
    delta_y = y[1] - y[0]

    for i in range(iterations):
        record.append([x0, y0])
        n, m = nearest(x0, y0, x, y)

        if Z[n, m] > 10000:
            print("result diverged.")
            sys.exit()

        f_x = (Z[n + 1, m] - Z[n - 1, m]) / (2 * delta_x)
        f_y = (Z[n, m + 1] - Z[n, m - 1]) / (2 * delta_y)

        f_xx = (Z[n + 1, m] - 2 * Z[n, m] + Z[n - 1, m]) / (delta_x ** 2)
        f_yy = (Z[n, m + 1] - 2 * Z[n, m] + Z[n, m - 1]) / (delta_y ** 2)
        f_xy = (Z[n + 1, m + 1] - Z[n - 1, m + 1] - Z[n + 1, m - 1] + Z[n - 1, m - 1]) / (4 * delta_x * delta_y)

        hessian = [[f_xx, f_xy], [f_xy, f_yy]]
        gradient = [f_x, f_y]
        hessian_inverse = np.linalg.inv(hessian)

        arr = alpha*np.matmul(hessian_inverse,gradient)

        x0 -= arr[0]
        y0 -= arr[1]

    record = np.asarray(record)

    print("Newton local extremum: {}, {}".format(x0, y0))

    return x0, y0, record

def gradient_descent(x, y, Z, x0, y0, alpha, iterations):
    record = []
    delta_x = x[1] - x[0]
    delta_y = y[1] - y[0]

    for i in range(iterations):
        # record.append([x0, y0])
        n, m = nearest(x0, y0, x, y)
        record.append([n - 100, m - 100])

        if Z[n, m] > 10000:
            print("result diverged.")
            sys.exit()

        f_x = (Z[n + 1, m] - Z[n - 1, m]) / (2 * delta_x)
        f_y = (Z[n, m + 1] - Z[n, m - 1]) / (2 * delta_y)

        gradient = [f_x, f_y]

        x0 -= alpha * gradient[0]
        y0 -= alpha * gradient[1]

    record = np.asarray(record)

    print("gradient descent local extremum: {}, {}".format(x0, y0))

    return x0, y0, record

if __name__ == "__main__":
    x = np.arange(-10, 10, .1)
    y = np.arange(-10, 10, .1)
    X, Y = np.meshgrid(x, y)

    Z = func(X, Y)

    x0, y0 = 7, -1.5

    x1, y1, record1 = newton(x, y, Z, x0, y0, 1, 20)
    x2, y2, record2 = gradient_descent(x, y, Z, x0, y0, 1, 20)

    # plt.imshow(Z, extent = [-100, 100, -100, 100])
    # plt.plot(record1[:,0], record1[:,1], 'go')
    # plt.plot(record2[:,0], record2[:,1], 'ro')
    # plt.show()

    #fig = plt.figure()
    #ax = fig.gca(projection='3d')
    # surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
    #                       cmap=cm.RdBu,linewidth=0, antialiased=False)

    # rx1, ry1 = np.meshgrid(record1[0], record2[1])
    # rx2, ry2 = np.meshgrid(record1[0], record2[1])
    #
    #
    # scatter1 = ax.scatter(rx1, ry1, func(rx1, ry1), c='r')
    # scatter2 = ax.scatter(rx2, ry2, func(rx2, ry2), c='r')
    #
    #
    # # print(record)
    #
    # plt.show()
