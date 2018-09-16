from __future__ import print_function
# I import this just for compatibility with python2 please use python3
# though.

import sys
# sys has useful utilities I need.

from time import clock as time_clock
# Time is being imported to measure
# running time for the factorize
# function.

import math
# Math is being imported to take the square root of n
# to set a largest possible endpoint to a prime

import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import interactive

import scipy.stats as stats

from error_in_s_o_l import er_in_sol


def get_mean(length):
    total = 0;
    for element in er_in_sol:
        total = total + element
    return total / length


def get_dev_from_mean(mean):
    set_of_deviations = []
    for element in er_in_sol:
        set_of_deviations.insert(0, (mean - element))
    return set_of_deviations


def square_deviations(set_of_devs):
    set_of_deviations_squared = []
    for element in set_of_devs:
        set_of_deviations_squared.insert(0, math.pow(element, 2))
    return set_of_deviations_squared


def sum_squared_deviations(squared_deviations):
    total = 0
    for element in squared_deviations:
        total = total + element
    return total


def investigate_histogram_in_np(random_vars):
    hist, bin_edges = np.histogram(random_vars)
    print('hist', hist)
    print('bin edges', bin_edges)


def build_histogram_in_matplotlib(random_vars):
    plt.figure(1)
    n, bins, patches = plt.hist(x=random_vars, bins='auto', color='#0504aa',
                                alpha=0.7, rwidth=0.85)
    plt.grid(axis='y', alpha=.75)
    plt.xlabel('value')
    plt.ylabel('frequency')
    plt.title('histogram of data')
    plt.text(23, 45, r'$\mu=15, b=3$')
    maxfreq = n.max()
    plt.ylim(ymax=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)
    interactive(True)
    plt.show()

def build_q_q_plot_using_stats(random_vars):
    plt.figure(2)
    z = ((random_vars-np.mean(random_vars))/np.std(random_vars))
    stats.probplot(z, dist='norm', plot=plt)
    plt.title('normal q-q plot')
    interactive(False)
    plt.show()


def calc_inferences_from_data():
    length = len(er_in_sol)
    mean = get_mean(length)
    deviation_from_mean_array = get_dev_from_mean(mean)
    squared_deviation_from_mean_array = square_deviations(deviation_from_mean_array)
    sum_of_squared_deviations = sum_squared_deviations(squared_deviation_from_mean_array)
    approx_avg_of_devs = sum_of_squared_deviations / (length - 1)
    standard_deviation = math.sqrt(approx_avg_of_devs)
    standard_error = standard_deviation / math.sqrt(length)
    random_var_set = [random.gauss(mean, standard_deviation) for _ in range(100)]
    # print('mean', mean)
    # print('sd: ', standard_deviation)
    # print('standard_error: ', standard_error)
    # print('new nums', random_var_set)
    # print('length', length)
    # investigate_histogram_in_np(random_var_set)
    build_histogram_in_matplotlib(random_var_set)
    build_q_q_plot_using_stats(random_var_set)


calc_inferences_from_data()
