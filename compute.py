
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
    total = 0;
    for element in squared_deviations:
        total = total + element;
    return total


def calc_inferences_from_data():
    length = len(er_in_sol)
    mean = get_mean(length)
    deviation_from_mean_array = get_dev_from_mean(mean)
    squared_deviation_from_mean_array = square_deviations(deviation_from_mean_array)
    sum_of_squared_deviations = sum_squared_deviations(squared_deviation_from_mean_array)
    approx_avg_of_devs = sum_of_squared_deviations/(length - 1)
    standard_deviation = math.sqrt(approx_avg_of_devs)
    standard_error = standard_deviation/math.sqrt(length)
    print('sd: ', standard_deviation)
    print('standard_error: ', standard_error)
    print('length', length)


calc_inferences_from_data()


