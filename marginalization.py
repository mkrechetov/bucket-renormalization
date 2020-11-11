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


NUM_STATES = 0

def extract_seattle_data(eps=1e-1, MU=300):
    # Read Data
    rawnumbers = pd.read_csv('./seattle/TractTravelRawNumbers.csv', header=None).values
    g_raw = rawnumbers/MU
    J_raw = np.log(1+np.exp(g_raw))/2

    # j_0 =
    # print(np.max(J_raw), np.min(J_raw))
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

    N = len(G.nodes)
    # eps = 1e-1
    # add edges
    for i in range(N):
        for j in range(N):
            if J_raw[j][i] > np.log(2)/2+eps:
                G.add_edge(i,j, weight=J_raw[j][i])
    return G

def ith_object_name(prefix, i):
    return prefix + str(int(i))
def ijth_object_name(prefix, i,j):
    return prefix + '(' + str(int(i)) + ',' + str(int(j)) + ')'

def generate_seattle(G, init_inf, inv_temp):
    model = GraphicalModel()
    N = len(G.nodes)
    # node_colors = ['b']*N

    for node in G.nodes:
        model.add_variable(ith_object_name('V', node))

    for a,b in G.edges:
        beta = G[a][b]['weight']
        log_values = np.array([beta, -beta, -beta, beta]).reshape([2,2])
        factor = Factor(
            name = ijth_object_name('F', a, b),
            variables = [ith_object_name('V', a), ith_object_name('V', b)],
            log_values = log_values)
        model.add_factor(factor)

    for node in G.nodes:
        # define factor definitions
        beta = -inv_temp #if node in init_inf else 0
        log_values = np.array([-beta, beta])
        factor = Factor(
            name = ith_object_name('B', node),
            variables = [ith_object_name('V', node)],
            log_values = log_values)
        model.add_factor(factor)

    # # this removes a variable and updates its neighbors MF.
    # for inf in init_inf:
    #     model.remove_variable(ith_object_name('V', inf))
    #
    #     adj_factors = model.get_adj_factors(ith_object_name('B', inf))
    #     model.remove_factors_from(adj_factors)
    #
    #     var_names = []
    #     for fac in adj_factors:
    #         for entry in fac.variables:
    #             var_names.append(entry.replace('V','B'))
    #     var_names = list(set(var_names))
    #
    #     nbrs = model.get_factors_from(var_names)
    #
    #     for nbr in nbrs:
    #         # if fac is None: continue
    #         fac = [f for f in factors if nbr.name.replace('B','') in f.name]
    #         if not fac: continue
    #         beta = nbr.log_values+fac[0].log_values[0]
    #         nbr.log_values = beta

    return model


def generate_star(N):
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
    return model


def extract_factor_matrix(model):
    J = np.zeros([N,N])
    for (a,b) in itertools.product(range(N), range(N)):
        if a < b:
            name = 'F({},{})'.format(a,b)
            fac = model.get_factor(name)
            J[a,b] = fac.log_values[1][1] if fac is not None else 0
            J[b,a] = J[a,b]
    return J

def extract_var_weights(model, nbr_num=-1):
    N = len(model.variables)
    H = np.zeros([N])
    for a in range(N):
        if a != nbr_num:
            name = 'B{}'.format(a)
            fac = model.get_factor(name)
            H[a] = fac.log_values[1]
    return H

def update_nbg_mf(model, init_inf):
    '''returns a copy of the GM modified by removing variable var'''

    for inf in init_inf:
        model.remove_variable(ith_object_name('V', inf))

        adj_factors = model.get_adj_factors(ith_object_name('B', inf))
        model.remove_factors_from(adj_factors)

        var_names = []
        for fac in adj_factors:
            for entry in fac.variables:
                var_names.append(entry.replace('V','B'))
        var_names = list(set(var_names))

        nbrs = model.get_factors_from(var_names)

        for nbr in nbrs:
            # if fac is None: continue
            fac = [f for f in factors if nbr.name.replace('B','') in f.name]
            if not fac: continue
            beta = nbr.log_values+fac[0].log_values[0]
            nbr.log_values = beta

def compute_partition_functions():
    filename = "seattle_marginal_Z_init_inf={}_BETA={}_MU={}_EPS={}.csv".format(init_inf, BETA, MU, eps)
    utils.append_to_csv(filename, ['Tract', 'Z_i', 'time'])

    Zi = []
    # N=len(seattle.variables)
    # H = extract_var_weights(seattle)
    node_buckets = [fac for fac in seattle.factors if 'B' in fac.name]
    # collect partition functions of modified GMs
    for index in range(N):
        # if index <= 61: continue
        var = seattle.variables[index]
        model_copy = seattle.copy()
        print('var {} has {} neighbors'.format(var, seattle.degree(var)))

        adj_factors = seattle.get_adj_factors(var)
        factors = [fac for fac in adj_factors if 'F' in fac.name]

        # remove variable and edges
        model_copy.remove_variable(var)
        model_copy.remove_factors_from(adj_factors)

        var_names = []
        for fac in adj_factors:
            for entry in fac.variables:
                var_names.append(entry.replace('V','B'))
        var_names = list(set(var_names))

        nbrs = model_copy.get_factors_from(var_names)

        for nbr in nbrs:
            # if fac is None: continue
            fac = [f for f in factors if nbr.name.replace('B','') in f.name]
            if not fac: continue
            beta = nbr.log_values+fac[0].log_values[0]
            nbr.log_values = beta
            # nbr_num = int(nbr.name.replace('B',''))

        H_temp = extract_var_weights(model_copy, index)
        print(H_temp)

        t1 = time.time()
        Z_copy = BucketRenormalization(model_copy, ibound=10).run(max_iter=1)
        t2 = time.time()
        print('partition function computation {} complete: {} (time taken: {})'.format(index, Z_copy, t2-t1))
        utils.append_to_csv(filename, [var, Z_copy, t2-t1])
        Zi.append(Z_copy)
    print(Zi)

def compute_marginal_probabilities(seattle):
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




# init_inf = [0, 81, 93]
init_inf = [0]
BETA = 3
MU = 300
eps = 1e-1
# for inf in init_inf:
#     for MU in MUS:
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
    quit()

def degree_distribution(seattle):
    '''degree distribution'''
    degree = [seattle.degree(var) for var in seattle.variables]
    plt.plot(range(N), degree)
    plt.title('degree of each node for eps = {}'.format(eps))
    plt.savefig('eps_treshold_{}_deg_dist_MU={}_BETA={}.png'.format(eps, MU, BETA))
    plt.show()
    # quit()

degree_distribution(seattle)

t1 = time.time()
Z = BucketRenormalization(seattle, ibound=10).run(max_iter=1)
t2 = time.time()
print('check 4')
print(Z)
print(t2-t1)
# quit()
H = extract_var_weights(seattle)
# print(H)
# quit()
normfac = np.exp(H)
# print(H)

# =====================================
compute_partition_functions()
# =====================================

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
