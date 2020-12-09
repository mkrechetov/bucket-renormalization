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


NUM_STATES = 0

# input into program from command line
parser = argparse.ArgumentParser()
parser.add_argument(
    '-m','--mu',
    default='100',
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
MU = int(args.mu)
eps = float(args.eps)

def extract_seattle_data(eps=1e-1, MU=300):
    # Read Data
    rawnumbers = pd.read_csv('./seattle/TractTravelRawNumbers.csv', header=None).values
    # g_raw = rawnumbers/MU
    # J_raw = np.log(1+np.exp(g_raw))/2

    # alternate way of estimating J_raw
    J_raw = -(rawnumbers/2)*np.log(1-MU)

    # j_0 =
    print(np.max(J_raw), np.min(J_raw))
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

    results=[]

    # UNCOMMENT THE NEXT LINE AND COMMENT THE THIRD ''' APPROXIMATLY 50 LINES BELOW TO RUN CODE SERIALLY
    #'''
    results.append(Parallel(n_jobs=mp.cpu_count())(delayed(compute_partition_functionsParallel)(index) for index in range(N)))
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



def compute_partition_functionsParallel(index):
    # I USED TRY/EXCEPT BECAUSE THE CODE FAILS BECAUSE A NUMBER GOES TO INFINITY WHILE CALCULATING
    try:
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
                var_names.append(entry.replace('V', 'B'))
        var_names = list(set(var_names))

        nbrs = model_copy.get_factors_from(var_names)

        for nbr in nbrs:
            # if fac is None: continue
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
        # utils.append_to_csv(filename, [var, Z_copy, t2 - t1])
        #Zi.append(Z_copy)
        return [var, Z_copy, t2 - t1]
    except Exception as e:
        print(e)
        traceback.print_exc()
        print("Failed on var: ",var)
        return []


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

def make_colormap(seq):
    """Return a LinearSegmentedColormap
    seq: a sequence of floats and RGB-tuples. The floats should be increasing
    and in the interval (0,1).
    """
    seq = [(None,) * 3, 0.0] + list(seq) + [1.0, (None,) * 3]
    cdict = {'red': [], 'green': [], 'blue': []}
    for i, item in enumerate(seq):
        if isinstance(item, float):
            r1, g1, b1 = seq[i - 1]
            r2, g2, b2 = seq[i + 1]
            cdict['red'].append([item, r1, r2])
            cdict['green'].append([item, g1, g2])
            cdict['blue'].append([item, b1, b2])
    return mcolors.LinearSegmentedColormap('CustomMap', cdict)

def drawProbabilityHeatmap(passedFileName,tractUVCoords,rawSeattleImage,probabilities):
    fig, ax = plt.subplots(figsize=(19.20,10.80))
    c = mcolors.ColorConverter().to_rgb
    rvb = make_colormap(
        [c('blue'), c('red')])
    norm = mpl.colors.Normalize(vmin=probabilities.T.min(), vmax=probabilities.T.max())

    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='3%', pad=0.05)

    mpl.colorbar.ColorbarBase(cax, cmap=rvb,
                              norm=norm,
                              orientation='vertical')

    ax.imshow(rawSeattleImage,cmap=rvb)
    for index in range(probabilities.shape[1]):
        ax.add_patch(Ellipse((tractUVCoords.iloc[index][1],tractUVCoords.iloc[index][2]), width=11, height=11,
                             edgecolor='None',
                             facecolor=(probabilities.iloc[0][index],0,1-probabilities.iloc[0][index],1),
                             linewidth=1))
    plt.tight_layout()
    plt.show()
    fig.savefig("./results/"+passedFileName+".png")#AUTOMATICALLY SAVES THE IMAGE FILES IN RESULTS FOLDER

def renormalizeProbability(input):
    output=pd.DataFrame()
    max_value=0
    min_value=1000000
    for i in range(input.shape[0]):  # TEST RANDOM PROBABILITIES
        if input.iloc[i][1]<min_value:
            min_value=input.iloc[i][1]
        if input.iloc[i][1]>max_value:
            max_value=input.iloc[i][1]
    for i in range(input.shape[0]):  # TEST RANDOM PROBABILITIES
        output[str(i)] = np.array([(input.iloc[i][1]-min_value)/(max_value-min_value)])
    return output


#\/\/\/ TESTING FOR SAVE PROBABILITY HEATMAP TO FILE
#TO DO: MIX THIS TEST CODE WITH HPC RELATED CODE.
# tractUVCoords = pd.read_csv('./seattle/tractUVCoordinates.csv')#GIS DATA WHICH SHOULD BE READ ONCE AND USED MULTIPLE TIMES
# rawSeattleImage=mpimg.imread('./seattle/SeattleRawImage.jpg')#GIS DATA WHICH SHOULD BE READ ONCE AND USED MULTIPLE TIMES

#TEST RANDOM PROBABILITIES. PROBABILITIES SHOULD BE IN A SINGLE ROW
# testProbabilities = pd.DataFrame()#TEST RANDOM PROBABILITIES
# for i in range(tractUVCoords.shape[0]):#TEST RANDOM PROBABILITIES
#     testProbabilities[str(i)] = rnd.rand(1)#TEST RANDOM PROBABILITIES
# testProbabilities = pd.read_csv('./results/seattle_marginal_probabilities_init_inf=[0]_BETA=3.0_MU=140_EPS=0.5.csv')#GIS DATA WHICH SHOULD BE READ ONCE AND USED MULTIPLE TIMES
# testProbabilities = renormalizeProbability(testProbabilities)
#A NAMING SCHEMA IS REQUIRED TO REPLACE "test". THE DIRECTORY IS SET INSIDE THE FUNCTION.
#tractUVCoords, AND rawSeattleImage SHOULD BE READ ONCE AND USED MULTIPLE TIMES.
# test_name = "seattle_marginal_probabilities_init_inf=[0]_BETA=3.0_MU=140_EPS=0.5"
# drawProbabilityHeatmap(test_name,tractUVCoords,rawSeattleImage,testProbabilities)
#^^^ TESTING FOR SAVE PROBABILITY HEATMAP TO FILE




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
    # quit()

def degree_distribution(seattle):
    '''degree distribution'''
    degree = [seattle.degree(var) for var in seattle.variables]
    weights = [G[i][j]['weight'] for i,j in G.edges ]
    maxJ = np.round(np.max(weights),3)
    minJ = np.round(np.min(weights),3)
    plt.plot(range(N), degree)
    plt.title('eps = {}, BETA = {}, MU = {},\n max J = {}, min J = {}'.format(eps, BETA, MU, maxJ, minJ))
    plt.savefig('./results/eps={}_MU={}_BETA={}_maxJ={}_minJ={}.png'.format(eps, MU, BETA, maxJ, minJ))
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
