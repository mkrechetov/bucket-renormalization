import numpy as np
import cvxpy as cp
import networkx as nx
import matplotlib.pyplot as plt
import itertools
import copy
import os
import json
import tools

N = np.genfromtxt('seattle_top20_travel_numbers.csv', delimiter=',')
h_as = [0.02, 0.05, 0.1, 0.2, 0.5]
#mus = [0.0000001,0.0000004,0.0000008,0.000001,0.000002,0.000004,0.000008,0.00001,0.00002,0.00003,0.00004,0.00005,0.00006,0.00007,0.00008,0.0001,0.00002,0.00004,0.00008,0.0001,0.00013,0.00016,0.00019,0.00022,0.00025,0.00027, 0.0003, 0.00033, 0.00036, 0.00037, 0.00038, 0.00039, 0.0004, 0.00043,0.00046,0.00049, 0.00051, 0.00054, 0.00057, 0.0006,0.0007,0.0008,0.0009,0.001]
musr = list(range(30))
mus = [mu*0.006/30 for mu in musr]

datadict = dict()

with open("Seattle20newfull2.json") as infile:
    datadict = json.load(infile)

for node in range(len(N)):
    infected = [node]
    for h_a in h_as:

        casedict = dict()
        for mu in mus:
            h = h_a * np.ones(len(N))
            cali = tools.get_cali(N, mu, h, infected, ibound=18, algorithm='gbr')
            cali.insert(node, 1)
            casedict["{0:0.7f}".format(mu)] = cali
        datadict[str((h_a, node))]['gbr18'] = casedict

        #casedict = dict()
        #for mu in mus:
        #    h = h_a * np.ones(len(N))
        #    cali = tools.get_cali(N, mu, h, infected, ibound=20, algorithm='gbr')
        #    cali.insert(node, 1)
        #    casedict["{0:0.7f}".format(mu)] = cali
        #datadict[str((h_a, node))]['gbr20'] = casedict

with open("Seattle20newfull3.json", "w") as outfile:
    json.dump(datadict, outfile)




