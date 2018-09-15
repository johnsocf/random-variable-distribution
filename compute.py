
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

def read_nums():
    total = 0;
    length = len(er_in_sol)
    for element in er_in_sol:
        total = total + element;
    mean = total/length;
    print('mean: ', mean)




read_nums()


