import os
import sys
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

def load_points_from_file(filename):
    """Loads 2d points from a csv called filename
    Args:
        filename : Path to .csv file
    Returns:
        (xs, ys) where xs and ys are a numpy array of the co-ordinates.
    """
    points = pd.read_csv(filename, header=None)
    return points[0].values, points[1].values


def view_data_segments(xs, ys, a, b, c, d, e, f):
    """Visualises the input file with each segment plotted in a different colour.
    Args:
        xs : List/array-like of x co-ordinates.
        ys : List/array-like of y co-ordinates.
    Returns:
        None
    """
    #Checks that there are equivalent number of x and y values
    assert len(xs) == len(ys)
    #Checks that data points are in sets of size 20
    assert len(xs) % 20 == 0
    #sets len_data variable to the number of x values
    len_data = len(xs)
    #len_data divided by 20 and returns rounded down to nearest int (should already be an int tho as asserted previously)
    num_segments = len_data // 20
    #Creates array (colour) of which number segment the data values are a part of e.g. colour = [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2]
    colour = np.concatenate([[i] * 20 for i in range(num_segments)])
    #Sets the colour map to Dark2
    plt.set_cmap('Dark2')
    #Plots a scatter graph
    x = np.linspace(0, 30)
    y1 = f * x + e
    y2 = d * x**3 + c * x**2 +  b * x + a
    fig, ax = plt.subplots()
    ax.scatter(xs, ys, c=colour)
    # plt.plot(x,y1)
    # plt.plot(x,y2)
    plt.show()


def split_into_segment(xs, ys):
    assert len(xs) == len(ys)
    assert len(xs) % 20 == 0
    len_data = len(xs)
    x_segments = [xs[n:n+20] for n in range(0, len_data, 20)]
    y_segments = [ys[n:n+20] for n in range(0, len_data, 20)]
    return x_segments, y_segments

def linear_least_squares(xs, ys):
    X = np.column_stack((np.ones(xs.shape), xs))
    A = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(ys)
    return A

def non_linear_least_squares(xs, ys):
    X = np.column_stack((np.ones(xs.shape), xs, xs**2, xs**3))
    A = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(ys)
    return A

def split_data(xs, ys):
    training = []
    validation = []
    x_training, x_validation = xs[:14], xs[14:]
    y_training, y_validation = ys[:14], ys[14:]
    training.append(x_training)
    training.append(y_training)
    validation.append(x_validation)
    validation.append(y_validation)
    return training, validation

def cross_validation(xs, ys):
    training, validation = split_data(xs, ys)
    error = ((Y_test - Yh_test)**2).mean()
    return 


# def sum_squared_error(xs, ys):



filename = sys.argv[1]
x, y = load_points_from_file(filename)
xs, ys = split_into_segment(x, y)
training, validation = split_data(x, y)
print(training)
print(validation)

# e, f = linear_least_squares(xs[0], ys[0])
# a, b, c, d = non_linear_least_squares(xs[2], ys[2])
# view_data_segments(x,y, a, b, c, d, e, f)