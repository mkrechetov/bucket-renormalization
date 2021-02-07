import numpy as np
import networkx as nx
import csv
import time
import sys
import argparse
import os
import random
import protocols
import utils
import copy
import json

sys.path.extend(["graphical_model/"])

from factor import Factor
from graphical_model import GraphicalModel

def ith_object_name(prefix, i):
    return prefix + str(int(i))

def ijth_object_name(prefix, i, j):
    return prefix + "(" + str(int(i)) + "," + str(int(j)) + ")"

def pandemic_model(J, h, infected):

    healthy_nodes = list(set(range(len(J))).difference(set(infected)))
    new_J = np.zeros((len(healthy_nodes), len(healthy_nodes)))
    for i in range(len(healthy_nodes)):
        for j in range(i + 1, len(healthy_nodes)):
            new_J[i, j] = J[healthy_nodes[i], healthy_nodes[j]]
            new_J[j, i] = J[healthy_nodes[j], healthy_nodes[i]]

    new_h = np.zeros(len(healthy_nodes))
    for i in range(len(healthy_nodes)):
        new_h[i] = h[healthy_nodes[i]]
        for j in range(len(infected)):
            new_h[i] -= J[healthy_nodes[i], infected[j]]

    graph = nx.from_numpy_matrix(new_J)
    model = GraphicalModel()
    model_size = len(graph.nodes())

    for i in range(model_size):
        model.add_variable(ith_object_name("V", i))

    for i in range(model_size):
        for j in range(i + 1, model_size):
            beta = new_J[i, j]
            log_values = np.array([beta, -beta, -beta, beta]).reshape([2, 2])
            factor = Factor(
                name=ijth_object_name("F", i, j),
                variables=[ith_object_name("V", i), ith_object_name("V", j)],
                log_values=log_values,
            )
            model.add_factor(factor)

    for i in range(model_size):
        factor = Factor(
            name=ith_object_name("B", i),
            variables=[ith_object_name("V", i)],
            log_values=np.array([new_h[i], -new_h[i]]),
        )
        model.add_factor(factor)

    return model

def get_cali_diff(N, mu, h, infected, ibound=10, algorithm='gbr', delta=0.00001):
    inference_protocol = protocols.inference_protocol_dict[algorithm]

    J = N * np.log(1 / (1 - mu))
    model = pandemic_model(J, h, infected)
    # true_logZ = model_protocol["true_inference"](model)

    # compute PF
    if inference_protocol["use_ibound"]:
        alg = inference_protocol["algorithm"](model, ibound)
    else:
        alg = inference_protocol["algorithm"](model)

    logZ = alg.run(**inference_protocol["run_args"])

    # compute CALI
    yet_healthy = list(set(range(len(J))).difference(set(infected)))
    CALI = []

    for node in yet_healthy:
        h_node = copy.copy(h)
        h_node[node] += delta
        model = pandemic_model(J, h_node, infected)


        if inference_protocol["use_ibound"]:
            alg = inference_protocol["algorithm"](model, ibound)
        else:
            alg = inference_protocol["algorithm"](model)

        logZ_node = alg.run(**inference_protocol["run_args"])

        CALI.append(-1 * (logZ_node - logZ) / delta)

    return CALI

def CALI(p):
    cali = []
    for prob in p:
        cali.append(2 * prob - 1)
    return cali

def get_cali(N, mu, h, infected, ibound=15, algorithm='gbr'):
    inference_protocol = protocols.inference_protocol_dict[algorithm]

    J = N * np.log(1 / (1 - mu))
    model = pandemic_model(J, h, infected)

    # compute PF
    if inference_protocol["use_ibound"]:
        alg = inference_protocol["algorithm"](model, ibound)
    else:
        alg = inference_protocol["algorithm"](model)

    logZ = alg.run(**inference_protocol["run_args"])

    # compute marginals
    yet_healthy = list(set(range(len(J))).difference(set(infected)))
    p_infected = []

    for node in yet_healthy:
        this_infected = copy.copy(infected)
        this_infected.append(node)
        model = pandemic_model(J, h, this_infected)

        if inference_protocol["use_ibound"]:
            alg = inference_protocol["algorithm"](model, ibound)
        else:
            alg = inference_protocol["algorithm"](model)

        # logZ_node = alg.run(**ip["run_args"])
        logZ_node = alg.run(**inference_protocol["run_args"])
        p_infected.append(np.exp(J[0, node]) * np.exp(-h[node]) * np.exp(logZ_node - logZ))

    return CALI(p_infected)

#datadict = dict()

#with open("Seattle20newfull2.json") as infile:
#    datadict = json.load(infile)

#keys= list(datadict.keys())
#d = datadict['(0.02, 0)']
#print(d['exact'])
#print(d['gbr15'])

#N = np.genfromtxt('seattle_top20_travel_numbers.csv', delimiter=',')
#h_as = [0.02, 0.05, 0.1, 0.2, 0.5]
#mus = [0.0000001,0.0000004,0.0000008,0.000001,0.000002,0.000004,0.000008,0.00001,0.00002,0.00003,0.00004,0.00005,0.00006,0.00007,0.00008,0.0001,0.00002,0.00004,0.00008,0.0001,0.00013,0.00016,0.00019,0.00022,0.00025,0.00027, 0.0003, 0.00033, 0.00036, 0.00037, 0.00038, 0.00039, 0.0004, 0.00043,0.00046,0.00049, 0.00051, 0.00054, 0.00057, 0.0006,0.0007,0.0008,0.0009,0.001]
#musr = list(range(30))
#mus = [mu*0.006/30 for mu in musr]

#cali1 = get_cali_diff(N, 0.001, 0.02 * np.ones(len(N)), [0], ibound=15, algorithm='gbr')
#print(cali1)

#cali2 = get_cali(N, 0.001, 0.02 * np.ones(len(N)), [0], ibound=15, algorithm='gbr')
#print(cali2)