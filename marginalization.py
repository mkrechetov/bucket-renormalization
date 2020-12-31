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
from testing import *

# input into program from command line
parser = argparse.ArgumentParser()
parser.add_argument(
    '-m','--mu',
    default='0.01',
    help='MU parameter - max interaction factor')
parser.add_argument(
    '--magfield',
    default='0.1',
    help='H_a paremeter - magnetic field')
parser.add_argument(
    '-t','--tau',
    default='-1',
    help='TAU parameter - minimal number of people traversing an edge')
parser.add_argument(
    '--seed',
    type=int,
    default=0)
parser.add_argument(
    '--testing',
    default=0,
    help='testing various aspects of the algorithm'
)
args = parser.parse_args()

np.random.seed(args.seed)
random.seed(args.seed)

# init_inf = [0, 81, 93]
init_inf = [52]
H_a = float(args.magfield)
MU = float(args.mu)
TAU = float(args.tau)
TESTING = int(args.testing)


# TESTING Purposes
if TESTING:
    testing_partition_function_dependence_on_TAU()
    testing_run_time()


print('experiment: init_inf={} H_a={} MU={} TAU={}'.format(init_inf, H_a, MU, TAU))

print('extracting seattle data...')
G = extract_seattle_data(TAU, MU)


print('generating GM...')
# generates a Graphical Model of Infection
# - removes and modifies GM to reflect initial seed of infection
# - sets magnetic field to all nodes to H_a
seattle = generate_seattle(G, init_inf, H_a)

# compute partition functions for Seattle GM and all sub-GMs
# =====================================
t1 = time.time()
compute_marginals(seattle, (init_inf, H_a, MU, TAU))
t2 = time.time()
# =====================================

print("compute_partition_functions RUNTIME: ",t2-t1)
