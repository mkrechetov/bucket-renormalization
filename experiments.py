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

parser = argparse.ArgumentParser()
parser.add_argument(
    '-m', '--model-type',
    default='grid_gmi',
    help='type of graphical model')
parser.add_argument(
    '-alg', '--algorithms',
    nargs = '+',
    default=['mf', 'bp', 'mbe', 'wmbe', 'mbr', 'gbr'],
    help = 'algorithms to be tested')
parser.add_argument(
    '--seed',
    type=int,
    default=0)
args = parser.parse_args()

np.random.seed(args.seed)
random.seed(args.seed)

model_protocol = protocols.model_protocol_dict[args.model_type]
print(model_protocol)
inference_protocols = [
    protocols.inference_protocol_dict[name]
    for name in args.algorithms]

def example_run_model(size = 15, delta = 1, ibound = 10):
    model = model_protocol['generator'](size, delta)
    tic = time.time()
    true_logZ = model_protocol['true_inference'](model)
    toc = time.time()
    print("time to compute true_logZ = {}".format(toc-tic))
    for ip in inference_protocols:
        if ip['name'] == 'GBR':
            alg = ip['algorithm'](model, ibound)

            tic = time.time()
            logZ = alg.run(**ip['run_args'])
            err = np.abs(true_logZ - logZ)
            toc = time.time()
    print('true_logZ = {}'.format(true_logZ))
    print('logZ = {}'.format(logZ))
    print('err = {}'.format(err))
    print('run time = {}'.format(toc-tic))
# example_run_model()

def grid_experiment(m,n):
    global model_protocol, inference_protocols
    # nb_size = 15
    delta = 1
    ibound = 10
    file_name = ('complexity[model={}_ibound={:d}_delta={:d}_griddim[m={:d}_n={:d}]].csv'.format(
        args.model_type, ibound, delta, m, n))
    utils.append_to_csv(file_name, ['size', 'alg','BE', 'BE time', 'approx Z', 'alg time', 'error'])

    model = model_protocol['generator'](m, n, delta)
    print(model.summary())
    tic = time.time()
    true_logZ = model_protocol['true_inference'](model)
    toc = time.time()
    true_logZ_time = toc-tic
    for ip in inference_protocols:
        if ip['name'] in ['MBR','GBR']:
            # edit the model to account for the initially infected node.
            alg = ip['algorithm'](model, ibound)

            tic = time.time()
            logZ = alg.run(**ip['run_args'])
            err = np.abs(true_logZ - logZ)
            toc = time.time()

            print('Alg: {:15}, Error: {:15.4f}, Time: {:15.2f}'.format(
                ip['name'], err, toc-tic))

            utils.append_to_csv(file_name, [m*n, ip['name'], true_logZ, true_logZ_time, logZ, toc-tic, err])
            print('experiment for {} nodes complete'.format(m*n))

def complete_graph_experiment(n, ibound = 10):
    global model_protocol, inference_protocols
    # nb_size = 15
    delta = 1
    # ibound = 10
    file_name = ('complexity[model={}_ibound={:d}_delta={:d}_complete[n={:d}]].csv'.format(
        args.model_type, ibound, delta, n))
    utils.append_to_csv(file_name, ['size', 'alg','BE', 'BE time', 'approx Z', 'alg time', 'error'])

    model = model_protocol['generator'](n, delta)
    # print(model.summary())
    tic = time.time()
    true_logZ = model_protocol['true_inference'](model) # computed using BE
    toc = time.time()
    true_logZ_time = toc-tic
    for ip in inference_protocols:
        if ip['name'] in ['MBR','GBR']:
            # edit the model to account for the initially infected node.
            alg = ip['algorithm'](model, ibound)

            tic = time.time()
            logZ = alg.run(**ip['run_args'])
            err = np.abs(true_logZ - logZ)
            toc = time.time()

            print('Alg: {:15}, Error: {:15.4f}, Time: {:15.2f}'.format(
                ip['name'], err, toc-tic))

            utils.append_to_csv(file_name, [n, ip['name'], true_logZ, true_logZ_time, logZ, toc-tic, err])
            print('experiment for {} nodes complete'.format(n))

def illustration_of_inference():
    G = nx.Graph()

    n = 5
    node_labels = range(n)
    edge_labels = []
    pos = {}
    for i in node_labels:
        G.add_node(i)
        pos[i] = [np.cos(2*np.pi*i/n), np.sin(2*np.pi*i/n)]

    for i in range(n):
        for j in range(i+1,n):
            G.add_edge(i,j, weight=np.random.uniform(0,1))
    nx.draw(G, pos)
    plt.show()

def marginalization(variable_name):
    global model_protocol, inference_protocols
    n = 5
    delta = 1
    model = model_protocol['generator'](n, delta)
    true_logZ = model_protocol['true_inference'](model) # computed using BE
    print(true_logZ)
    # marginalize over a given variable
    for factor in model.factors:
        if variable_name in factor.variables:
            print(factor, variable_name in factor.variables)
            fac = factor.marginalize([variable_name], inplace=True)
            print(fac)
            continue
    true_logZ = model_protocol['true_inference'](model) # computed using BE
    print(true_logZ)



marginalization('V0')

# illustration_of_inference()
# grid_experiment(4,5)
# grid_experiment(5,10)
# grid_experiment(7,7)
# grid_experiment(10,10)
# grid_experiment(10,20)
# grid_experiment(14,14)
# grid_experiment(15,15)
# grid_experiment(16,16)
# grid_experiment(17,17)
# grid_experiment(18,18)
# grid_experiment(19,19)
# grid_experiment(20,20)

# complete_graph_experiment(10)
# complete_graph_experiment(12)
# complete_graph_experiment(14)
# complete_graph_experiment(16)
# complete_graph_experiment(18)
# complete_graph_experiment(20)
# complete_graph_experiment(22)
# complete_graph_experiment(24)
# complete_graph_experiment(26)
# complete_graph_experiment(28)
# complete_graph_experiment(30)

# complete_graph_experiment(50)
# complete_graph_experiment(100)
# complete_graph_experiment(200)
# complete_graph_experiment(10)
# complete_graph_experiment(10)
# complete_graph_experiment(10)
# complete_graph_experiment(10)
