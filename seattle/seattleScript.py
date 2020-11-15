import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import os

TractData = []
for file_name in os.listdir('./'):
    if 'Tract' in file_name:
        TractData.append(file_name)
print(TractData)

probabilities = pd.read_csv(TractData[1], header=None)
summary = pd.read_csv(TractData[0])
summary

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

# nx.draw(G, pos)

for key, val in pos.items():
    print(key, val)

import sys
sys.path.extend(['../'])
from gmi import GMI


simulation = GMI(G)

simulation.dynamic_process(view=1,externalView=1,isSync=1)