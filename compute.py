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
from scipy.stats import norm

import scipy.stats as stats

from error_in_s_o_l import er_in_sol
from r_a_d import g_counts, g_time


def get_mean_for_continuous(data, length):
    total = 0;
    for element in data:
        total = total + element
    return total / length

def get_mean_for_discrete(data_set_timeline, probability_set, length):
    mean_sum = 0
    for indx, element in enumerate(data_set_timeline):
        mean_sum = mean_sum + (data_set_timeline([indx] * probability_set))
    return (mean_sum/ length)


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


def build_histogram_in_matplotlib(data, title, figure_num, interactive_bool):
    plt.figure(figure_num)
    n, bins, patches = plt.hist(x=data, bins='auto', color='#0504aa',
                                alpha=0.7, rwidth=0.85)
    plt.grid(axis='y', alpha=.75)
    plt.xlabel('value')
    plt.ylabel('frequency')
    plt.title(title)
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
    mean = get_mean_for_continuous(data, length)
    deviation_from_mean_array = get_dev_from_mean(mean)
    squared_deviation_from_mean_array = square_deviations(deviation_from_mean_array)
    sum_of_squared_deviations = sum_squared_deviations(squared_deviation_from_mean_array)
    approx_avg_of_devs = sum_of_squared_deviations / (length - 1)
    standard_deviation = math.sqrt(approx_avg_of_devs)
    standard_error = standard_deviation / math.sqrt(length)
    return mean, standard_deviation


def plot_pdf(data_set):
    # ???
    fig, ax = plt.subplots(1, 1)
    x = np.linspace(norm.ppf(0.01), norm.ppf(0.99), 100)
    ax.plot(data_set, norm.pdf(x), 'r-', lw=5, alpha=0.6, label='norm pdf')
    rv = norm()
    ax.plot(data_set, rv.pdf(x), 'k-', lw=2, label='frozen pdf')
    vals = norm.ppf([0.001, 0.5, 0.999])
    np.allclose([0.001, 0.5, 0.999], norm.cdf(vals))
    r = norm.rvs(size=1000)
    ax.hist(r, density=True, histtype='stepfilled', alpha=0.2)
    ax.legend(loc='best', frameon=False)
    interactive(False)
    plt.show()

def build_plots_for_speed_of_light(initial_data_set, random_var_built_data_set):
    build_q_q_plot_using_stats(initial_data_set, 'q q plot of original data', 1, True)

    plot_pdf(random_var_built_data_set)
    build_histogram_in_matplotlib(random_var_built_data_set, 'histogram for random variable set speed of light', 2, True)
    build_q_q_plot_using_stats(random_var_built_data_set, 'q q plot of random var', 3, False)

def set_up_speed_of_light_dist():
    mean, standard_d = calc_inferences_from_data(er_in_sol)
    random_var_set_speed_of_light = [random.gauss(mean, standard_d) for _ in range(100)]
    build_plots_for_speed_of_light(er_in_sol, random_var_set_speed_of_light)

def set_up_geiger_dist():
    # X = # geiser counts over time.
    length = len(g_counts)

    # Construct a random variable that models the geiger counts **

    # get probability dist for geiser...
    # stub
    probability = [0]


    # get mean and standard d
    mean = get_mean_for_discrete(g_time, probability, length)

    # build data set based using rand based on mean and sd?

    # build pdf:
    build_histogram_in_matplotlib(g_counts, 'histogram for geiger count', 4, True)

    # build pdf
    # build q q plot (similar to existing function) + (overlay on histogram)


def init():
    set_up_speed_of_light_dist()
    #set_up_geiger_dist()


init()
