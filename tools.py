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
# 1) create graph with all nodes degree 2 with largest numbers
# 2) use global threshold to add more factors to each variable
# for each census tract
 # collect 2-10 most significant travels # thresholding parameter to exclude edges
 # least degree is 2
 # overall threshhold  T = 100
 # check all nodes > T
 # populate with J_raw
 # check J_raw is not 0 or infty
 # generate GM

# check degree dist
def extract_seattle_data(TAU=100, MU=300):
    # Read Data
    data = pd.read_csv('./seattle/TractTravelRawNumbers.csv', header=None).values
    # g_raw = rawnumbers/MU
    # J_raw = np.log(1+np.exp(g_raw))/2

    # alternate way of estimating J_raw
    J_raw = -(data/2)*np.log(1-MU)

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
            if idx in [0,1] or data[count][val] > TAU:
                G.add_edge(count, val, weight=J_raw[count][val])
        count+=1
    return G

def ith_object_name(prefix, i):
    return prefix + str(int(i))
def ijth_object_name(prefix, i,j):
    return prefix + '(' + str(int(i)) + ',' + str(int(j)) + ')'

def generate_seattle(G, init_inf, inv_temp):
    model = GraphicalModel()

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
        beta = -inv_temp if node in init_inf else 0
        log_values = np.array([-beta, beta])
        factor = Factor(
            name = ith_object_name('B', node),
            variables = [ith_object_name('V', node)],
            log_values = log_values)
        model.add_factor(factor)

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

'''def update_nbg_mf(model, init_inf):
    ''''''returns a copy of the GM modified by removing variable var''''''

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
            nbr.log_values = beta'''

def degree_distribution(seattle, G, params):
    '''degree distribution'''
    degree = [seattle.degree(var) for var in seattle.variables]
    weights = [G[i][j]['weight'] for i,j in G.edges ]
    # counts, bins = np.histogram(weights)
    # plt.hist(weights, bins=100)
    plt.title('min value = {}'.format(np.min(weights)))
    maxJ = np.round(np.max(weights),3)
    minJ = np.round(np.min(weights),3)
    N = len(G.nodes)
    plt.plot(range(N), degree)
    plt.title(R"$\tau$ = {}, $\beta$ = {}, $\mu$ = {}," "\n" "max J = {}, min J = {}".format(params[0], params[1], params[2], maxJ, minJ))
    plt.savefig('./results/TAU={}_MU={}_BETA={}_maxJ={}_minJ={}.png'.format(params[0], params[1], params[2], maxJ, minJ))
    plt.show()
    # quit()

def compute_partition_functions(seattle, init_inf, params):
    BETA, MU, TAU = params
    filename = "seattle_marg_Z_init_inf={}_BETA={}_MU={}_TAU={}.csv".format(init_inf, BETA, MU, TAU)
    utils.append_to_csv(filename, ['Tract', 'Z_i', 'time'])

    Zi = []
    N = len(seattle.variables)
    # H = extract_var_weights(seattle)
    node_buckets = [fac for fac in seattle.factors if 'B' in fac.name]

    results=[]

    # UNCOMMENT THE NEXT LINE AND COMMENT THE THIRD ''' APPROXIMATLY 50 LINES BELOW TO RUN CODE SERIALLY
    #'''
    results.append(Parallel(n_jobs=mp.cpu_count())(delayed(compute_partition_functionsParallel)(seattle, index) for index in range(N)))
    '''
    results.append([])
    # collect partition functions of modified GMs
    for index in range(N):
        # I USED TRY/EXCEPT BECAUSE THE CODE FAILS BECAUSE A NUMBER GOES TO INFINITY WHILE CALCULATING
        try:
            #if index != 26: continue
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
            results[0].append([var, Z_copy, t2 - t1])
            #utils.append_to_csv(filename, [var, Z_copy, t2-t1])
            #Zi.append(Z_copy)
        except Exception as e:
            print(e)
            traceback.print_exc():
            print("Failed on var: ", var)
            results[0].append([])
    '''
    # COMMENT THE ABOVE ''' TO RUN CODE SERIALLY
    for index in range(N):
        try:# I USED TRY/EXCEPT BECAUSE THE CODE FAILS BECAUSE A NUMBER GOES TO INFINITY WHILE CALCULATING
            Zi.append(results[0][index][1])
        except:# IF PREVIOUS CODES FAIL FILL Zi WITH 0
            print("Zi ", index, " not calculated!")
            Zi.append(0)

    print(Zi)
    for index in range(N):
        try:# I USED TRY/EXCEPT BECAUSE THE CODE FAILS BECAUSE A NUMBER GOES TO INFINITY WHILE CALCULATING
            utils.append_to_csv(filename, [results[0][index][0],results[0][index][1],results[0][index][2]])
        except:# IF PREVIOUS CODES FAIL WRITE ZERO FOR THE V VALUES
            utils.append_to_csv(filename, ["V"+str(index), 0, 0])
            print("Row ",index," written by 0!")



def compute_partition_functionsParallel(seattle, index):
    # I USED TRY/EXCEPT BECAUSE THE CODE FAILS BECAUSE A NUMBER GOES TO INFINITY WHILE CALCULATING
    try:
        var = seattle.variables[index]
        model_copy = seattle.copy()
        # print('var {} has {} neighbors'.format(var, seattle.degree(var)))

        adj_factors = seattle.get_adj_factors(var)
        factors = [fac for fac in adj_factors if 'F' in fac.name]

        # remove variable and edges
        model_copy.remove_variable(var)
        model_copy.remove_factors_from(adj_factors)

        # collect variable names of neighbors
        var_names = []
        for fac in adj_factors:
            for entry in fac.variables:
                var_names.append(entry.replace('V', 'B'))
        var_names = list(set(var_names))

        nbrs = model_copy.get_factors_from(var_names)

        # update the magnetic field of neighboring variables
        for nbr in nbrs:
            fac = [f for f in factors if nbr.name.replace('B', '') in f.name]
            if not fac: continue
            beta = nbr.log_values + fac[0].log_values[0]
            nbr.log_values = beta
            # nbr_num = int(nbr.name.replace('B',''))

        H_temp = extract_var_weights(model_copy, index)
        # print(H_temp)

        t1 = time.time()
        Z_copy = BucketRenormalization(model_copy, ibound=10).run(max_iter=1)
        t2 = time.time()
        print('partition function computation {} complete: {} (time taken: {})'.format(index, Z_copy, t2 - t1))
        return [var, Z_copy, t2 - t1]
    except Exception as e:
        print(e)
        traceback.print_exc()
        print("Failed on var: ",var)
        return []


def compute_marginal_probabilities(seattle):
    filename = "seattle_marginal_probabilities_init_inf={}_BETA={}_MU={}_TAU={}.csv".format(init_inf, BETA, MU, TAU)
    utils.append_to_csv(filename, ['Tract', 'probability'])

    pfs = utils.read_csv("seattle_marginal_Z_init_inf={}_BETA={}_MU={}_TAU={}.csv".format(init_inf, BETA, MU, TAU))
    Zi = [float(entry[1]) for entry in pfs[1:]]
    print(Zi)

    P = lambda i: normfac[i]*Zi[i]/Z


    for idx in range(N):
        marg_prob = P(idx)
        print('P( x_{} = {} ) = {}'.format(idx, 1, marg_prob))
        utils.append_to_csv(filename, [seattle.variables[idx], marg_prob])


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
