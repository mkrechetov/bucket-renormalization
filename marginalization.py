import numpy as np
import csv
import time
import sys
sys.path.extend(['./', './seattle/'])
import argparse
import os
import random
import protocols
import utils
import networkx as nx
import matplotlib.pyplot as plt
from gmi import GMI
import pandas as pd
from graphical_model import *
from factor import *
from generate_model import generate_complete_gmi, generate_complete
from bucket_elimination import BucketElimination
from bucket_renormalization import BucketRenormalization
import itertools
import traceback

from joblib import Parallel, delayed
import multiprocessing as mp
import matplotlib.image as mpimg
import matplotlib as mpl
from matplotlib.patches import Ellipse
import numpy.random as rnd
import matplotlib.colors as mcolors
from mpl_toolkits.axes_grid1 import make_axes_locatable
from tools import *

# input into program from command line
parser = argparse.ArgumentParser()
parser.add_argument(
    '-m','--mu',
    default='0.01',
    help='MU parameter - max interaction factor')
parser.add_argument(
    '-b','--beta',
    default='3',
    help='BETA paremeter - inverse temperature')
parser.add_argument(
    '-t','--tau',
    default='100',
    help='TAU parameter - minimal number of people traversing an edge')
parser.add_argument(
    '--seed',
    type=int,
    default=0)
args = parser.parse_args()

np.random.seed(args.seed)
random.seed(args.seed)

# init_inf = [0, 81, 93]
init_inf = [0]
BETA = float(args.beta)
MU = float(args.mu)
TAU = float(args.tau)


print('init_inf={} MU={} BETA={} TAU={}'.format(init_inf, MU, BETA, TAU))

G = extract_seattle_data(TAU, MU)

seattle = generate_seattle(G, init_inf, BETA)
print(seattle.summary())

degree_distribution(seattle, G, (TAU, BETA, MU))


# compute partition function for Seattle GM
# =====================================
# t1 = time.time()
# Z = BucketRenormalization(seattle, ibound=10).run(max_iter=1)
# t2 = time.time()
# =====================================


# compute partition functions for Seattle GM and all sub-GMs
# =====================================
t1 = time.time()
compute_marginals(seattle, init_inf, (BETA, MU, TAU))
t2 = time.time()
# =====================================

print("compute_partition_functions RUNTIME: ",t2-t1)


# filename = "seattle_marg_prob_init_inf={}_BETA={}_MU={}_TAU={}.csv".format(init_inf, BETA, MU, TAU)
# utils.append_to_csv(filename, ['Tract', 'probability'])

# pfs = utils.read_csv("seattle_marg_Z_init_inf={}_BETA={}_MU={}_TAU={}.csv".format(init_inf, BETA, MU, TAU))
# Zi = [float(entry[1]) for entry in pfs[1:]]

# magnetic field H
# H = extract_var_weights(seattle)
# normfac = np.exp(H)
# formula to compute marginal probabilities
# P = lambda i: normfac[i]*Zi[i]/Z

# compute marginal probabilities
# for idx in range(N):
    # marg_prob = P(idx)
    # print('P( x_{} = {} ) = {}'.format(idx, 1, marg_prob))
    # utils.append_to_csv(filename, [seattle.variables[idx], marg_prob])
