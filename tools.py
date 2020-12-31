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

def extract_seattle_data(TAU=-1, MU=300):
    # Read Data
    data = pd.read_csv('./seattle/TractTravelRawNumbers.csv', header=None).values
    # g_raw = rawnumbers/MU
    # J_raw = np.log(1+np.exp(g_raw))/2

    # alternate way of estimating J_raw
    J_raw = -data*np.log(1-MU)

    # MU = 1-np.exp(-J_raw/np.max(data))

    minJ = np.round(np.min(J_raw), 5)
    maxJ = np.round(np.max(J_raw), 5)

    summary = pd.read_csv('./seattle/TractSummary159.csv')
    # Create Graph
    G = nx.Graph()
    pos = {}
    # add nodes
    for row in summary.iterrows():
        G.add_node(row[0],
                   Tract=row[1]['Tract'],
                   Sus=row[1]['Sus'],
                   Inf=row[1]['Inf'],
                   Symp=row[1]['Symp'],
                   RecoveredCalc=row[1]['RecoveredCalc'],
                   Lat=row[1]['Lat'],
                   Lon=row[1]['Lon']
                  )
        pos[row[0]] = [row[1]['Lat'], row[1]['Lon']]

    # add edges
    count = 0
    for row in data:
        # sort the number of people in descending order and collect their indices
        indices = np.argsort(row)[::-1]

        for idx, val in enumerate(indices):
            # if the first two numbers, or numbers greater than threshold
            if idx in [0,1] or data[count][val] > TAU: # threshold criteria
                G.add_edge(count, val, weight=J_raw[count][val])
        count+=1
    return G

def ith_object_name(prefix, i):
    return prefix + str(int(i))
def ijth_object_name(prefix, i,j):
    return prefix + '(' + str(int(i)) + ',' + str(int(j)) + ')'

def generate_seattle(G, init_inf, H_a):
    '''
    This is done in 3 steps:
        1) add all variable names to the GM
        2) add all factors to the GM
        3) add 1-factors to each variable
    '''
    # A collection of factors
    model = GraphicalModel()

    for node in G.nodes:
        model.add_variable(ith_object_name('V', node))

    # define factors
    for a,b in G.edges:
        Jab = G[a][b]['weight']
        log_values = np.array([Jab, -Jab, -Jab, Jab]).reshape([2,2])
        factor = Factor(
            name = ijth_object_name('F', a, b),
            variables = [ith_object_name('V', a), ith_object_name('V', b)],
            log_values = log_values)
        model.add_factor(factor)

    # define magentic fields
    for node in G.nodes:
        h_a = H_a
        log_values = np.array([-h_a, h_a])
        factor = Factor(
            name = ith_object_name('B', node),
            variables = [ith_object_name('V', node)],
            log_values = log_values)
        model.add_factor(factor)

    # Modify the graph by conditioning it on the initially infected nodes
    # removes the initially infected nodes too.
    for var in init_inf:
        update_MF_of_neighbors_of(model, ith_object_name('V', var))

    return model

def get_neighbor_factors_of(model, var):
    '''
        Returns a list of B-factors of a given variable var in the GM model
    '''
    adj_factors = model.get_adj_factors(var) # adj_factors contains a B factor and neighboring F factors
    factors = [fac for fac in adj_factors if 'F' in fac.name] # factors contains only F factors

    # collect variable names of neighbors
    var_names = []
    for fac in factors:
        for entry in fac.variables:
            var_names.append(entry.replace('V', 'B'))
    # this is a set e.g. ['B52', 'B81', 'B0']
    var_names = list(set(var_names))

    # remove the factor associated to var
    var_names.remove(var.replace('V', 'B'))

    # collection of B factors
    nbrs = model.get_factors_from(var_names)

    return nbrs

def update_MF_of_neighbors_of(model, var):
    '''
        This modifies the GM 'copy' by
        1) removing the var variable and associated edges
        2) updating the magnetic field of neighbors of variable var
    '''
    # print('getting neighbors of {}'.format(var))
    # print('node {} has {} neighbors'.format(var, model.degree(var)-1))

    # get B-factors of the neighboring nodes (excluding index)
    nbrs = get_neighbor_factors_of(model, var)

    adj_factors = model.get_adj_factors(var) # adj_factors contains a B factor and neighboring F factors
    factors = [fac for fac in adj_factors if 'F' in fac.name] # factors contains only F factors

    # print('removing {} and associated neighbors'.format(var))

    # update the magnetic field of neighboring variables
    for nbr in nbrs:
        # collect the pair-wise factor containing the neighbor's bucket name
        fac = [f for f in factors if nbr.name.replace('B', '') in f.name].pop()
        # update the neighbor's magentic field
        nbr.log_values -= fac.log_values[1]
        # print("updated neighbor {} log value to {}".format(nbr.name, nbr.log_values))

    # remove variable var and associated factors
    model.remove_variable(var)
    model.remove_factors_from(adj_factors)


def compute_PF_of_modified_GM(model, index):
    '''
        Each computation done in parallel consists of
        1) removing the index variable and associated edges
        2) updating the neighbors' magnetic field
        3) compute the partition function of the modified GM
    '''
    var = model.variables[index] # collect the ith variable name
    copy = model.copy()

    # this modifies the GM copy by
    # removing var and factors connected to it
    # updating the neighbors' magnetic fields
    update_MF_of_neighbors_of(copy, var)
    try:
        # compute partition function of the modified GM
        t1 = time.time()
        Z_copy = BucketRenormalization(copy, ibound=10).run(max_iter=1)
        t2 = time.time()
        print('partition function computation {} complete: {} (time taken: {})'.format(index, Z_copy, t2 - t1))
        return [var, Z_copy, t2 - t1]
    except Exception as e:
        print(e)
        return []


def compute_marginals(model, params):
    init_inf, H_a, MU, TAU = params

    # ==========================================
    # Compute partition function for Seattle GM
    # ==========================================
    try:
        t1 = time.time()
        Z = BucketRenormalization(model, ibound=10).run(max_iter=1)
        t2 = time.time()
        print('partition function = {}'.format(Z))
        print('time taken for GBR = {}'.format(t2-t1))
    except Exception as e:
        raise Exception(e)
    # ==========================================


    N = len(model.variables)

    results=[]
    results.append(
        Parallel(n_jobs=mp.cpu_count())(delayed(compute_PF_of_modified_GM)(model, index) for index in range(N))
    )

    # collect partition functions of sub-GMs
    Zi = []
    for index in range(N):
        Zi.append(results[0][index][1])

    # compute marginal probabilities formula conditioned on initial seed
    # ==========================================
    P = lambda i: Zi[i]/Z
    # ==========================================

    # write data to file
    filename = "seattle_marg_prob_init_inf={}_H_a={}_MU={}_TAU={}.csv".format(init_inf, H_a, MU, TAU)
    utils.append_to_csv(filename, ['Tract', 'Z_i', 'time', 'P_i'])
    for index in range(N):
        marg_prob = P(index)
        print('P( x_{} = {} ) = {}'.format(index, 1, marg_prob))
        utils.append_to_csv(filename, [results[0][index][0],results[0][index][1],results[0][index][2], marg_prob])
    utils.append_to_csv(filename, ['whole GM',Z,t2-t1, 'N/A'])


def degree_distribution(model, G, params):
    '''degree distribution'''
    H_a, MU, TAU = params
    degree = [model.degree(var) for var in model.variables]
    weights = [G[i][j]['weight'] for i,j in G.edges]
    # counts, bins = np.histogram(weights)
    # plt.hist(weights, bins=100)
    plt.title('min value = {}'.format(np.min(weights)))
    maxJ = np.round(np.max(weights),3)
    minJ = np.round(np.min(weights),3)
    N = len(G.nodes)
    plt.plot(range(N), degree)
    plt.title(R"$\beta$ = {}, $\mu$ = {}, $\tau$ = {}," "\n" "max J = {}, min J = {}".format(H_a, MU, TAU, maxJ, minJ))
    plt.savefig('./results/H_a={}_MU={}_TAU={}_maxJ={}_minJ={}.png'.format(H_a, MU, TAU, maxJ, minJ))
    plt.show()
    # quit()




'''def generate_star(N):
    model = GraphicalModel()

    center = 0

    for i in range(N):
        model.add_variable(ith_object_name('V', i))
        val = np.random.uniform(0,1)
        log_values = [-val, val]
        # print(log_values)
        bucket = Factor(
            name = ith_object_name('B', i),
            variables = [ith_object_name('V', i)],
            log_values = np.array([-val, val]))
        model.add_factor(bucket)

        if i != center:
            beta = np.random.uniform(0,1)
            log_values = np.array([beta, -beta, -beta, beta]).reshape([2,2])
            factor = Factor(
                name = ijth_object_name('F', center, i),
                variables = [ith_object_name('V', center), ith_object_name('V', i)],
                log_values = log_values)
            model.add_factor(factor)
    return model'''


'''def extract_factor_matrix(model):
    J = np.zeros([N,N])
    for (a,b) in itertools.product(range(N), range(N)):
        if a < b:
            name = 'F({},{})'.format(a,b)
            fac = model.get_factor(name)
            J[a,b] = fac.log_values[1][1] if fac is not None else 0
            J[b,a] = J[a,b]
    return J'''

'''def extract_var_weights(model, nbr_num=-1):
    N = len(model.variables)
    # store magnetic fields
    H = np.zeros([N])
    for a in range(N):
        # you can skip neighbor nbr_num
        if a == nbr_num: continue
        # get the B factor
        fac = model.get_factor('B{}'.format(a))
        # store the beta value
        H[a] = fac.log_values[0] # collects positive beta
    return H'''
