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
from generate_model import generate_complete_gmi, generate_seattle
from bucket_elimination import BucketElimination
from bucket_renormalization import BucketRenormalization
import itertools


NUM_STATES = 0

def extract_seattle_data():
    # Read Data
    probabilities = pd.read_csv('./seattle/TractTravelProbabilities.csv', header=None)
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
    # add edges
    for i in range(N):
        for j in range(N):
            if probabilities[j][i] != 0:
                G.add_edge(i,j, weight=probabilities[j][i])
    return G

def extract_factor_matrix(model):
    J = np.zeros([N,N])
    for (a,b) in itertools.product(range(N), range(N)):
        if a < b:
            name = 'F({},{})'.format(a,b)
            fac = model.get_factor(name)
            J[a,b] = fac.log_values[1][1] if fac is not None else 0
            J[b,a] = J[a,b]
    return J

def extract_var_weights(model):
    H = np.zeros([N])
    for a in range(N):
        name = 'B{}'.format(a)
        fac = model.get_factor(name)
        H[a] = fac.log_values[1]
    return H


# delta = 1
init_inf = [0]
inv_temp = 10
# model = generate_complete_gmi(N, delta, init_inf, inv_temp)
G = extract_seattle_data()
seattle = generate_seattle(G, init_inf, inv_temp)
N = len(seattle.variables)
Z = BucketRenormalization(seattle, ibound=10).run(max_iter=1)
H = extract_var_weights(seattle)
factors = np.exp(H)
Zi = []
count = 0

filename = "seattle_marginal_Z_init_inf={}.csv".format(init_inf)
utils.append_to_csv(filename, ['Tract', 'Z_i'])
# print(seattle.factors[:4])
# quit()
num_factors = []
for var in seattle.variables:
    if count == 81:
        copy = seattle.copy()
        print('copy #{} made.'.format(count))
        # print('factor list length:')
        num_factors.append(len([fac for fac in copy.factors if var in fac.variables]))
        factors81 = [fac for fac in copy.factors if var in fac.variables]
        # print(factors81)
        # print(factors81[:3])
        # quit()
        copy.contract_variable(var)
        print('variable {} contracted'.format(var))
        Z_copy = BucketRenormalization(copy, ibound=10).run(max_iter=0)
        Zi.append(Z_copy)
        print('partition function computation {} complete: {}'.format(count, Z_copy))
        utils.append_to_csv(filename, [var, Z_copy])
    count +=1
# print(Zi)
# plt.plot(range(len(num_factors)),num_factors)
# plt.title('Number of factors associated to each tract')
# plt.xlabel('Tract number')
# plt.ylabel('number of factors')
#
# plt.show()

filename = "seattle_marginal_probabilities_init_inf={}.csv".format(init_inf)
utils.append_to_csv(filename, ['Tract', 'probability'])

P = lambda i: factors[i]*Zi[i]/Z
for idx in range(N):
    marg_prob = P(idx)
    print('P( x_{} = {} ) = {}'.format(idx, [1, marg_prob]))
    utils.append_to_csv(filename, [seattle.variables[idx], marg_prob])
