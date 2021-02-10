import numpy as np
import json
import tools
import multiprocessing as mp

CPUs = mp.cpu_count()

N = np.genfromtxt('seattle_top20_travel_numbers.csv', delimiter=',')
h_as = [0.02, 0.05, 0.1, 0.2, 0.5]
#mus = [0.0000001,0.0000004,0.0000008,0.000001,0.000002,0.000004,0.000008,0.00001,0.00002,0.00003,0.00004,0.00005,0.00006,0.00007,0.00008,0.0001,0.00002,0.00004,0.00008,0.0001,0.00013,0.00016,0.00019,0.00022,0.00025,0.00027, 0.0003, 0.00033, 0.00036, 0.00037, 0.00038, 0.00039, 0.0004, 0.00043,0.00046,0.00049, 0.00051, 0.00054, 0.00057, 0.0006,0.0007,0.0008,0.0009,0.001]
musr = list(range(30))
mus = [mu*0.006/30 for mu in musr]

datadict = dict()

with open("Seattle20newfull2.json") as infile:
    datadict = json.load(infile)

for node in range(len(N)):
    for h_a in h_as:
        h = h_a * np.ones(len(N))

        #casedict = dict()
        #params = [{'N': N, 'mu': mus[i], 'h': h, 'infected': [node], 'ibound': 18, 'algorithm': 'gbr'} for i in range(len(mus))]
        #with mp.Pool(CPUs) as p:
        #    calis = p.map(tools.get_cali, params)
        #for i in range(len(mus)):
        #    casedict["{0:0.7f}".format(mus[i])] = calis[i]
        #datadict[str((h_a, node))]['gbr18'] = casedict

        casedict = dict()
        params = [(N, mus[i], h,  [node],  15, 'gbr') for i in range(len(mus))]
        processes = []

        for i in range(len(mus)):
            p = mp.Process(target=tools.get_cali, args=params[i])
            processes.append(p)
            p.start()

        calis = []
        for p in processes:
            calis.append(p.join())

        for i in range(len(mus)):
            casedict["{0:0.7f}".format(mus[i])] = calis[i]
        datadict[str((h_a, node))]['gbr15'] = casedict

with open("Seattle20newfull5.json", "w") as outfile:
    json.dump(datadict, outfile)




