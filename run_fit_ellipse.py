#!/usr/bin/env python2.7
''' run_fit_ellipse.py by Nicky van Foreest '''
import numpy as np
from numpy.random import rand as rand
from fit_ellipse import *


# square a number
def sq(x):
    return x * x


# rotation matrix
def rotation_matrix(theta):
    st = np.sin(theta)
    ct = np.cos(theta)
    return np.matrix([[ct, st], [st, ct]])


# test the fitting on randomly generated ellipse data
n_samples = 40
for sample_i in range(0, n_samples):
    R = np.arange(0, 2. * np.pi, 0.01)
    n = len(R)

    def rand_d(n):
        return .1 + 1.5 * rand() - .1 * rand() * rand(n)
    # random ellipse data
    x_0, y_0, y_s, x_s = rand_d(n), rand_d(n), rand_d(n), rand_d(n)
    x = x_0 + x_s * np.cos(R) + .01*rand(n)
    y = y_0 + y_s * np.sin(R) + .01*rand(n)

    # random rotation
    theta = rand() * np.pi * 2.
    rot = rotation_matrix(theta)

    # apply rotation matrix
    for i in range(0, n):
        xy = np.matrix([x[i], y[i]]).T
        xy = np.dot(rot, xy)
        x[i], y[i] = xy[0], xy[1]

    # fit an ellipse to the above data
    a, b, center0, center1, phi = fit_ellipse(x, y)
    center, axes = (center0, center1), (a, b)

    # generate points on the fitted ellipse
    a, b = axes
    xx = center[0] + a * np.cos(R) * np.cos(phi) - b * np.sin(R) * np.sin(phi)
    yy = center[1] + a * np.cos(R) * np.sin(phi) + b * np.sin(R) * np.cos(phi)

    # plot the data points and the fitted ellipse
    plt.figure(0)
    plt.plot(x, y, color='blue', label='points')
    plt.plot(xx, yy, '+', color='red', label='fitted ellipse', linewidth=2.)
    plt.legend()
    plt.axes().set_aspect('equal', 'datalim')
    plt_fn = 'plot' + str(sample_i) + '.png'
    plt.savefig(plt_fn)
    print "+w", plt_fn
    plt.clf()
