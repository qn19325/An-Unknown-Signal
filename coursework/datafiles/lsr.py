import os
import sys
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

def load_points_from_file(filename):
    points = pd.read_csv(filename, header=None)
    return points[0].values, points[1].values
    
def view_data_segments(xs, ys, a, f):
    assert len(xs) == len(ys)
    assert len(xs) % 20 == 0
    len_data = len(xs)
    num_segments = len_data // 20
    colour = np.concatenate([[i] * 20 for i in range(num_segments)])
    plt.set_cmap('Dark2')
    _, x_segment, y_segment = split_into_segment(xs, ys)
    fig, ax = plt.subplots()
    ax.scatter(x_segment, y_segment, c=colour)
    for idx, val in enumerate(x_segment):
        x = np.linspace(min(x_segment[idx]),max(x_segment[idx]))
        coefficients = a[idx]
        if f[idx] == 0:
            y = coefficients[1] * x + coefficients[0]
            plt.plot(x, y)
        elif f[idx] == 1:
            y = coefficients[3] * x**3 + coefficients[2] * (x**2) + coefficients[1] * x + coefficients[0]
            plt.plot(x, y)
        elif f[idx] == 2:
            y = coefficients[1] * np.sin(x) + coefficients[0]
            plt.plot(x,y)
    plt.show()

def split_into_segment(xs, ys):
    assert len(xs) == len(ys)
    assert len(xs) % 20 == 0
    len_data = len(xs)
    num_segments = len_data // 20
    x_segments = [xs[n:n+20] for n in range(0, len_data, 20)]
    y_segments = [ys[n:n+20] for n in range(0, len_data, 20)]
    return num_segments, x_segments, y_segments

### LEAST SQUARES ###
def linear_least_squares(xs, ys):
    X = np.column_stack((np.ones(xs.shape), xs))
    A = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(ys)
    return A

def non_linear_least_squares(xs, ys):
    order = 3
    X = np.ones(xs.shape)
    for i in range(order):
        x = xs**(i+1)
        X = np.column_stack((X, x))
    A = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(ys)
    return A

def unknown_func_least_squares(xs, ys):
    X = np.column_stack((np.ones(xs.shape), np.sin(xs)))
    A = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(ys)
    return A

### SUM SQUARED ERROR ###
def sum_squared_error(f, coefficients, xs, ys):
    sum_square_error = 0
    if f == 0:
        for idx, val in enumerate(xs):
            fitted_y = coefficients[1] * val + coefficients[0]
            actual_y = ys[idx]
            sum_square_error = sum_square_error + ((actual_y - fitted_y)**2)
    elif f == 1:
        for idx, val in enumerate(xs):
            fitted_y = coefficients[3] * (val**3) + coefficients[2] * (val**2) + coefficients[1] * (val) + coefficients[0]
            actual_y = ys[idx]
            sum_square_error = sum_square_error + ((actual_y - fitted_y)**2)
    elif f == 2:
        for idx, val in enumerate(xs):
            fitted_y = coefficients[1] * np.sin(val) + coefficients[0]
            actual_y = ys[idx]
            sum_square_error = sum_square_error + ((actual_y - fitted_y)**2)
    return sum_square_error

### CROSS VALIDATION ###
def cross_validation(xs, ys, k):
    data = np.column_stack((xs, ys))
    # np.random.shuffle(data)
    avg = len(data) / float(k)
    split_data = []
    last = 0.0

    while last < len(data):
        split_data.append(data[int(last):int(last + avg)])
        last += avg
    linear_error = []
    non_linear_error = []
    unknown_error = []
    for i in range(len(split_data)):
        test = split_data[i]
        training = []
        for j in range(len(split_data)):
            if i != j:
                training.extend(split_data[j])
        training = np.vstack(training)

        linear_ls = linear_least_squares(training[:,0], training[:,1])
        non_linear_ls = non_linear_least_squares(training[:,0], training[:,1])
        unknown_ls = unknown_func_least_squares(training[:,0], training[:,1])

        linear_Cv = sum_squared_error(0, linear_ls, test[:,0], test[:,1])
        non_linear_Cv = sum_squared_error(1, non_linear_ls, test[:,0], test[:,1])
        unknown_Cv = sum_squared_error(2, unknown_ls, test[:,0], test[:,1])

        linear_error.append(linear_Cv)
        non_linear_error.append(non_linear_Cv)
        unknown_error.append(unknown_Cv)

    return linear_error, non_linear_error, unknown_error
    



filename = sys.argv[1]
x, y = load_points_from_file(filename)
num_segments, xs, ys = split_into_segment(x, y)
total_error = 0
coefficients = []
func_types = []
for i in range(num_segments):
    linear_ls = linear_least_squares(xs[i], ys[i])
    non_linear_ls = non_linear_least_squares(xs[i], ys[i])
    unknown_ls = unknown_func_least_squares(xs[i], ys[i])

    linear_errors, non_linear_errors, unknown_errors = cross_validation(xs[i], ys[i], 20)

    linear_Cv = np.mean(linear_errors)
    non_linear_Cv = np.mean(non_linear_errors)
    unknown_Cv = np.mean(unknown_errors)
    
    if(linear_Cv < non_linear_Cv and linear_Cv < unknown_Cv):
        coefficients.append(linear_ls)
        func_types.append(0)
        sse = sum_squared_error(0, linear_ls, xs[i], ys[i])
        total_error = total_error +  sse
    elif(non_linear_Cv < linear_Cv and non_linear_Cv < unknown_Cv):
        coefficients.append(non_linear_ls)
        func_types.append(1)
        sse = sum_squared_error(1, non_linear_ls, xs[i], ys[i])
        total_error = total_error + sse
    else:
        coefficients.append(unknown_ls)
        func_types.append(2)
        sse = sum_squared_error(2, unknown_ls, xs[i], ys[i])
        total_error = total_error + sse

print('Total Error -', total_error)
if(len(sys.argv)==3):
      if(sys.argv[2] == '--plot'):
          view_data_segments(x, y, coefficients, func_types)