import numpy as np
import json
import tools
import multiprocessing as mp
import argparse
import os

CPUs = mp.cpu_count()

def get_method_dict(N, hrange, murange, algorithm, ibound):
    method_dict = dict()

    for node in range(len(N)):
        for h_a in hrange:
            h = h_a * np.ones(len(N))

            case_dict = dict()
            params = [(N, murange[i], h,  [node],  15, 'gbr') for i in range(len(murange))]
            processes = []

            for i in range(len(murange)):
                p = mp.Process(target=tools.get_cali, args=params[i])
                processes.append(p)
                p.start()

            calis = []
            for p in processes:
                calis.append(p.join())

            for i in range(len(murange)):
                case_dict["{0:0.7f}".format(murange[i])] = calis[i]


            method_dict[str((h_a, node))][str(algorithm)+str(ibound)] = case_dict

    return method_dict

hrange = [0.02, 0.05, 0.1, 0.2, 0.5]
algorithms = [('bp', None), ('mf', None), ('gbr', 15), ('gbr', 18), ('gbr', 20)]

parser = argparse.ArgumentParser()
parser.add_argument("--data", help="path to data-file")
parser.add_argument("--muNum", default=30, help="number of MUs", type=int)
parser.add_argument("-muMax", 0.006, help="last value of MU", tupe=float)
parser.add_argument("-n", default='dict.json', help="dictionary name")
args = parser.parse_args()

N = np.genfromtxt(args.data, delimiter=',')
musr = list(range(args.muMax))
mus = [mu*args.muMax/args.muNum for mu in musr]

datadict = dict()

for alg in algorithms:
    datadict[str(alg[0])+str(alg[1])] = get_method_dict(N, hrange, mus, alg[0], alg[1])

with open(args.n, "w") as outfile:
    json.dump(datadict, outfile)