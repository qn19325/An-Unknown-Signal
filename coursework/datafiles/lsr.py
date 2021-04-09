import os
import sys
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

def load_points_from_file(filename):
    points = pd.read_csv(filename, header=None)
    return points[0].values, points[1].values
    
def view_data_segments(xs, ys, a):
    print(a)
    assert len(xs) == len(ys)
    assert len(xs) % 20 == 0
    len_data = len(xs)
    num_segments = len_data // 20
    colour = np.concatenate([[i] * 20 for i in range(num_segments)])
    plt.set_cmap('Dark2')
    x = np.linspace(0,30)
    if len(a) == 2:
        y = a[1] * x + a[0]
    elif len(a) == 4:
        y = a[3] * x**3 + a[2] * x**2 +  a[1] * x + a[0]
    fig, ax = plt.subplots()
    ax.scatter(xs, ys, c=colour)
    plt.plot(x, y)
    plt.show()

def split_into_segment(xs, ys):
    assert len(xs) == len(ys)
    assert len(xs) % 20 == 0
    len_data = len(xs)
    num_segments = len_data // 20
    x_segments = [xs[n:n+20] for n in range(0, len_data, 20)]
    y_segments = [ys[n:n+20] for n in range(0, len_data, 20)]
    return num_segments, x_segments, y_segments

def linear_least_squares(xs, ys):
    X = np.column_stack((np.ones(xs.shape), xs))
    A = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(ys)
    return A

def non_linear_least_squares(xs, ys):
    X = np.column_stack((np.ones(xs.shape), xs, xs**2, xs**3))
    A = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(ys)
    return A

def sum_squared_error(coefficients, xs, ys):
    sum_squared_error = 0
    if len(coefficients) == 2:
        for idx, val in enumerate(xs):
            fitted_y = coefficients[1] * val + coefficients[0]
            actual_y = ys[idx]
            sum_squared_error = sum_squared_error + ((actual_y - fitted_y)**2)
    elif len(coefficients) == 4:
        for idx, val in enumerate(xs):
            fitted_y = coefficients[3] * (val**3) + coefficients[2] * (val**2) + coefficients[1] * (val) + coefficients[0]
            actual_y = ys[idx]
            sum_squared_error = sum_squared_error + ((actual_y - fitted_y)**2)
    return sum_squared_error



filename = sys.argv[1]
x, y = load_points_from_file(filename)
num_segments, xs, ys = split_into_segment(x, y)
A = linear_least_squares(x, y)
B = non_linear_least_squares(x, y)
print(A,B)
linear_sse = sum_squared_error(A, x, y)
non_linear_sse = sum_squared_error(B, x, y)
if(linear_sse < non_linear_sse):
    view_data_segments(x, y, A)
else:
    view_data_segments(x, y, B)

# e, f = linear_least_squares(xs[0], ys[0])
# a, b, c, d = non_linear_least_squares(xs[2], ys[2])
# view_data_segments(x,y, a, b, c, d, e, f)