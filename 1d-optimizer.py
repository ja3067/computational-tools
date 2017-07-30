import numpy as np
import matplotlib.pyplot as plt
import sys

def func(x):
    return np.sin(x)

def nearest(x, X):
    return np.abs(x - X).argmin()

def newton(x, f, x0, alpha, iterations):
    record = []
    delta_x = x[1] - x[0]

    for i in range(iterations):
        record.append(x0)
        n = nearest(x0, x)

        if f[n] > 10000:
            print("result diverged.")
            sys.exit()

        deriv = (f[n + 1] - f[n - 1]) / (2 * delta_x)

        second_deriv = (f[n + 1] - 2 * f[n] + f[n - 1]) / (delta_x ** 2)

        x0 -= alpha * deriv / second_deriv

    record = np.asarray(record)

    print("Newton local extremum: {}".format(x0))

    return x0, record

def gradient_descent(x, f, x0, alpha, iterations):
    record = []
    delta_x = x[1] - x[0]

    for i in range(iterations):
        record.append(x0)
        n = nearest(x0, x)

        if f[n] > 10000:
            print("result diverged.")
            sys.exit()

        deriv = (f[n + 1] - f[n - 1]) / (2 * delta_x)

        x0 -= alpha * deriv

    record = np.asarray(record)

    print("gradient descent local extremum: {}".format(x0))

    return x0, record

if __name__ == "__main__":
    x = np.arange(-10, 10, .01)

    f = func(x)

    x0 = 5

    x1, record1 = newton(x, f, x0, 1, 20)
    x2, record2 = gradient_descent(x, f, x0, 1, 20)

    plt.plot(x, f)
    plt.plot(record1, func(record1), 'g>', record2, func(record2), 'rx')
    plt.show()