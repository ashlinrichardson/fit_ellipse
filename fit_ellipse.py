#!/usr/bin/env python2.7
''' fit_ellipse.py by Nicky van Foreest.

Many thanks to Andrew G. Sund for his contribution:
he recommended to implement with arctan2 (and % for float).
'''
import os
import numpy as np
from numpy.linalg import svd
from numpy.linalg import inv
import matplotlib.pyplot as plt


def ellipse_center(a):
    """@brief calculate ellipse centre point

    @param a the result of __fit_ellipse
    """
    b, c, d, f, g, a = a[1] / 2, a[2], a[3] / 2, a[4] / 2, a[5], a[0]
    num = b * b - a * c
    x0 = (c * d - b * f) / num
    y0 = (a * f - b * d) / num
    return np.array([x0, y0])


def ellipse_axis_length(a):
    """@brief calculate ellipse axes lengths

    @param a the result of __fit_ellipse
    """
    b, c, d, f, g, a = a[1] / 2, a[2], a[3] / 2, a[4] / 2, a[5], a[0]
    up = 2 * (a * f * f + c * d * d + g * b * b - 2 * b * d * f - a * c * g)
    down1 = (b * b - a * c) *\
            ((c - a) * np.sqrt(1 + 4 * b * b / ((a - c) * (a - c))) - (c + a))
    down2 = (b * b - a * c) *\
            ((a - c) * np.sqrt(1 + 4 * b * b / ((a - c) * (a - c))) - (c + a))
    res1, res2 = np.sqrt(up / down1), np.sqrt(up / down2)
    return np.array([res1, res2])


def ellipse_angle_of_rotation(a):
    """@brief calculate ellipse rotation angle

    @param a the result of __fit_ellipse
    """
    b, c, d, f, g, a = a[1] / 2, a[2], a[3] / 2, a[4] / 2, a[5], a[0]
    if b == 0:
        return (0. if a > c else np.pi / 2.)
    else:
        return np.arctan2(2 * b, (a - c)) / 2


def __fit_ellipse(x, y):
    """@brief fit an ellipse to supplied data points
                (internal method.. use fit_ellipse below...)
    @param x first coordinate of points to fit (array)
    @param y second coord. of points to fit (array)
    """
    x, y = x[:, np.newaxis], y[:, np.newaxis]
    D = np.hstack((x * x, x * y, y * y, x, y, np.ones_like(x)))
    S, C = np.dot(D.T, D), np.zeros([6, 6])
    C[0, 2], C[2, 0], C[1, 1] = 2, 2, -1
    U, s, V = svd(np.dot(inv(S), C))
    return U[:, 0]


def fit_ellipse(x, y):
    """@brief fit an ellipse to supplied data points: the 5 params
        returned are:

        a - major axis length
        b - minor axis length
        cx - ellipse centre (x coord.)
        cy - ellipse centre (y coord.)
        phi - rotation angle of ellipse bounding box

    @param x first coordinate of points to fit (array)
    @param y second coord. of points to fit (array)
    """
    e = __fit_ellipse(x, y)
    centre, phi = ellipse_center(e), ellipse_angle_of_rotation(e)
    a, b = ellipse_axis_length(e)

    # assert that a is the major axis (otherwise swap and correct angle)
    if(b > a):
        tmp = b
        b = a
        a = tmp
        phi %= 2. * np.pi  # ensure angle in [0, 2pi]
    return [a, b, centre[0], centre[1], phi]
