import os
import sys
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

def load_points_from_file(filename):
    points = pd.read_csv(filename, header=None)
    return points[0].values, points[1].values
    
def view_data_segments(xs, ys, a):
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
        if len(coefficients) == 2:
            y = coefficients[1] * x + coefficients[0]
            plt.plot(x, y)
        elif len(coefficients) == 3:
            # y = coefficients[2] * np.sin(x) + coefficients[1] * np.cos(x) + coefficients[0]
            y = coefficients[2] * (x**2) + coefficients[1] * x + coefficients[0]
            plt.plot(x, y)
        elif len(coefficients) == 4:
            y = coefficients[3] * x**3 + coefficients[2] * x**2 + coefficients[1] * x + coefficients[0]
            plt.plot(x,y)
        else:
            y = coefficients[8] * (x**8) + coefficients[7] * x**7 + coefficients[6] * x**6 + coefficients[5] * x**5 + coefficients[4] * x**4 + coefficients[3] * x**3 + coefficients[2] * x**2 + coefficients[1] * x + coefficients[0]
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
    X = np.column_stack((np.ones(xs.shape), np.cos(xs), np.sin(xs)))
    A = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(ys)
    return A

def sum_squared_error(coefficients, xs, ys):
    sum_squared_error = 0
    if len(coefficients) == 2:
        for idx, val in enumerate(xs):
            fitted_y = coefficients[1] * val + coefficients[0]
            actual_y = ys[idx]
            sum_squared_error = sum_squared_error + ((actual_y - fitted_y)**2)
    elif len(coefficients) == 3:
        for idx, val in enumerate(xs):
            # fitted_y = coefficients[2] * np.sin(val) + coefficients[1] * np.cos(val) + coefficients[0]
            fitted_y = coefficients[2] * (val**2) + coefficients[1] * val + coefficients[0]
            actual_y = ys[idx]
            sum_squared_error = sum_squared_error + ((actual_y - fitted_y)**2)
    elif len(coefficients) == 4:
        for idx, val in enumerate(xs):
            fitted_y = coefficients[3] * (val**3) + coefficients[2] * (val**2) + coefficients[1] * (val) + coefficients[0]
            actual_y = ys[idx]
            sum_squared_error = sum_squared_error + ((actual_y - fitted_y)**2)
    else:
        for idx, val in enumerate(xs):
            fitted_y = coefficients[8] * (val**8) + coefficients[7] * (val**7) + coefficients[6] * (val**6) + coefficients[5] * (val**5) + coefficients[4] * (val**4) + coefficients[3] * (val**3) + coefficients[2] * (val**2) + coefficients[1] * (val) + coefficients[0]
            actual_y = ys[idx]
            sum_squared_error = sum_squared_error + ((actual_y - fitted_y)**2)
    return sum_squared_error

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

        linear_sse = sum_squared_error(linear_ls, test[:,0], test[:,1])
        non_linear_sse = sum_squared_error(non_linear_ls, test[:,0], test[:,1])
        unknown_sse = sum_squared_error(unknown_ls, test[:,0], test[:,1])

        linear_error.append(linear_sse)
        non_linear_error.append(non_linear_sse)
        unknown_error.append(unknown_sse)

    return linear_error, non_linear_error, unknown_error
    



filename = sys.argv[1]
x, y = load_points_from_file(filename)
num_segments, xs, ys = split_into_segment(x, y)
total_error = 0
coefficients = []
for i in range(num_segments):
    linear_ls = linear_least_squares(xs[i], ys[i])
    non_linear_ls = non_linear_least_squares(xs[i], ys[i])
    unknown_ls = unknown_func_least_squares(xs[i], ys[i])

    # linear_sse = sum_squared_error(linear_ls, xs[i], ys[i])
    # print(sum_squared_error(non_linear_ls, xs[i], ys[i]))
    # unknown_sse = sum_squared_error(unknown_ls, xs[i], ys[i])

    linear_errors, non_linear_errors, unknown_errors = cross_validation(xs[i], ys[i], 20)

    # print(linear_errors)
    # print(non_linear_errors)
    # print(unknown_errors)

    linear_sse = np.mean(linear_errors)
    non_linear_sse = np.mean(non_linear_errors)
    unknown_sse = np.mean(unknown_errors)

    # print(linear_sse)
    # print(non_linear_sse)
    # print(unknown_sse)
    
    if(linear_sse < non_linear_sse and linear_sse < unknown_sse):
        coefficients.append(linear_ls)
    elif(non_linear_sse < linear_sse and non_linear_sse < unknown_sse):
        coefficients.append(non_linear_ls)
    else:
        coefficients.append(unknown_ls)
    # coefficients.append(non_linear_ls)

    total_error = total_error + min(linear_sse, non_linear_sse, unknown_sse)
    # total_error = total_error + non_linear_sse
print(coefficients)
print('Total Error -', total_error)
view_data_segments(x, y, coefficients)