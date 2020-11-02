import numpy as np
import csv
import time
import sys
sys.path.extend(['./'])
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
from generate_model import generate_complete_gmi
from bucket_elimination import BucketElimination
import itertools


NUM_STATES = 0

def extract_factor_matrix(model):
    J = np.zeros([N,N])
    for (a,b) in itertools.product(range(N), range(N)):
        if a < b:
            name = 'F({},{})'.format(a,b)
            fac = model.get_factor(name)
            J[a,b] = fac.log_values[1][1]
    return J

def extract_var_weights(model):
    H = np.zeros([N])
    for a in range(N):
        name = 'B{}'.format(a)
        fac = model.get_factor(name)
        H[a] = fac.log_values[1]
    return H

def is_valid(state):

    for node in init_inf:
        if state[node] < 1: return 0

    for a in range(N):
        for b in range(N):
            varsigma = 1 if [a,b] in model.active else 0
            if state[b] < varsigma*state[a]: return 0

    return 1

def probability(state):
    global NUM_STATES

    # check if it is a valid state
    if not is_valid(state): return 0

    NUM_STATES+=1

    scale = np.sum(np.sum(J))
    interaction = np.dot(np.dot(J, state), state) #+scale
    var_potential = np.dot(H, state) #- inv_temp/2

    numerator = np.exp(interaction+var_potential)

    return numerator/Z

def marginal_prob(index, val=1):
    prob = 0
    for state in states:
        if state[index] == val:
            p = probability(state)
            prob += p
    return prob

N = 15
delta = 1
init_inf = [0]
inv_temp = 100
model = generate_complete_gmi(N, delta, init_inf, inv_temp)
Z = BucketElimination(model).run()
J = extract_factor_matrix(model)
H = extract_var_weights(model)

states = list(itertools.product([-1,1], repeat=N))

val = 1
for idx in range(N):
    p = marginal_prob(idx, 1)
    q = marginal_prob(idx, -1)
    # print(p,q, p+q)
    print('P( x_{} = {} ) = {}'.format(idx,val, p))
