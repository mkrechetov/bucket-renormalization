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
    '-e','--eps',
    default='.2',
    help='EPS parameter - interaction threshhold')
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
eps = float(args.eps)


print('init_inf={} MU={} BETA={} eps={}'.format(init_inf, MU, BETA, eps))

G = extract_seattle_data(eps, MU)

seattle = generate_seattle(G, init_inf, BETA)
N = len(seattle.variables)
print(seattle.summary())

def test0():
    nb_vars = 70
    delta = 1.0

    for i in range(2,20):
        print(10*i)
        toy = generate_complete(10*i, delta)
        # toy = generate_star(10*i)
        t1 = time.time()
        Z = BucketRenormalization(toy, ibound=10).run(max_iter=1)
        t2 = time.time()

        print(Z)
        print(t2-t1)
    # quit()

def degree_distribution(seattle):
    '''degree distribution'''
    degree = [seattle.degree(var) for var in seattle.variables]
    weights = [G[i][j]['weight'] for i,j in G.edges ]
    # counts, bins = np.histogram(weights)
    plt.hist(weights, bins=100)
    plt.title('min value = {}'.format(np.min(weights)))
    maxJ = np.round(np.max(weights),3)
    minJ = np.round(np.min(weights),3)
    # plt.plot(range(N), degree)
    # plt.title('eps = {}, BETA = {}, MU = {},\n max J = {}, min J = {}'.format(eps, BETA, MU, maxJ, minJ))
    # plt.savefig('./results/eps={}_MU={}_BETA={}_maxJ={}_minJ={}.png'.format(eps, MU, BETA, maxJ, minJ))
    plt.show()
    # quit()

degree_distribution(seattle)

t1 = time.time()
Z = BucketRenormalization(seattle, ibound=10).run(max_iter=1)
t2 = time.time()
# print('check 4')
print('partition function = {}'.format(Z))
print('time taken for GBR = {}'.format(t2-t1))
# quit()
H = extract_var_weights(seattle)
# print(H)
# quit()
normfac = np.exp(H)
# print(H)

t1 = time.time()
# =====================================
compute_partition_functions()
# =====================================
t2 = time.time()
print("compute_partition_functions RUNTIME: ",t2-t1)

# quit()
filename = "seattle_marginal_probabilities_init_inf={}_BETA={}_MU={}_EPS={}.csv".format(init_inf, BETA, MU, eps)
utils.append_to_csv(filename, ['Tract', 'probability'])

pfs = utils.read_csv("seattle_marginal_Z_init_inf={}_BETA={}_MU={}_EPS={}.csv".format(init_inf, BETA, MU, eps))
Zi = [float(entry[1]) for entry in pfs[1:]]
print(Zi)

P = lambda i: normfac[i]*Zi[i]/Z


for idx in range(N):
    marg_prob = P(idx)
    print('P( x_{} = {} ) = {}'.format(idx, 1, marg_prob))
    utils.append_to_csv(filename, [seattle.variables[idx], marg_prob])
