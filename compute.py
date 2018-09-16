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


def build_histogram_in_matplotlib(random_vars, figure_num, interactive_bool):
    plt.figure(figure_num)
    n, bins, patches = plt.hist(x=random_vars, bins='auto', color='#0504aa',
                                alpha=0.7, rwidth=0.85)
    plt.grid(axis='y', alpha=.75)
    plt.xlabel('value')
    plt.ylabel('frequency')
    plt.title('histogram of data')
    plt.text(23, 45, r'$\mu=15, b=3$')
    maxfreq = n.max()
    plt.ylim(ymax=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)
    interactive(interactive_bool)
    plt.show()

def build_q_q_plot_using_stats(data, title, figure_num, interactive_bool):
    plt.figure(figure_num)
    z = ((data-np.mean(data))/np.std(data))
    stats.probplot(z, dist='norm', plot=plt)
    plt.title(title)
    interactive(interactive_bool)
    plt.show()


def calc_inferences_from_data(data):
    length = len(data)
    mean = get_mean(length)
    deviation_from_mean_array = get_dev_from_mean(mean)
    squared_deviation_from_mean_array = square_deviations(deviation_from_mean_array)
    sum_of_squared_deviations = sum_squared_deviations(squared_deviation_from_mean_array)
    approx_avg_of_devs = sum_of_squared_deviations / (length - 1)
    standard_deviation = math.sqrt(approx_avg_of_devs)
    standard_error = standard_deviation / math.sqrt(length)
    return mean, standard_deviation


def build_plots_for_speed_of_light(initial_data_set, random_var_built_data_set):
    build_q_q_plot_using_stats(initial_data_set, 'q q plot of original data', 1, True)
    build_histogram_in_matplotlib(random_var_built_data_set, 2, True)
    build_q_q_plot_using_stats(random_var_built_data_set, 'q q plot of random var', 3, False)


def init():
    mean, standard_d = calc_inferences_from_data(er_in_sol)
    random_var_set_speed_of_light = [random.gauss(mean, standard_d) for _ in range(100)]
    build_plots_for_speed_of_light(er_in_sol, random_var_set_speed_of_light)


init()
