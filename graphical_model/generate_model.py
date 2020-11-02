import os
import sys
from copy import copy
from factor import Factor
from graphical_model import GraphicalModel
import numpy as np
from functools import reduce
sys.path.extend(['inference/'])
# import networkx as nx
# import matplotlib.pyplot as plt

file_dir_path = os.path.dirname(os.path.realpath(__file__))

def ith_object_name(prefix, i):
    return prefix + str(int(i))
def ijth_object_name(prefix, i,j):
    return prefix + '(' + str(int(i)) + ',' + str(int(j)) + ')'

def generate_complete(nb_vars, delta):
    model = GraphicalModel()
    interaction = delta * np.random.uniform(
        -1.0, 1.0, nb_vars*(nb_vars-1))
    bias = np.random.uniform(-0.1, 0.1, [nb_vars])

    for i in range(nb_vars):
        model.add_variable(ith_object_name('V', i))

    for i in range(nb_vars):
        for j in range(i+1, nb_vars):
            beta = interaction[i * nb_vars + j ]
            log_values = np.array([beta, -beta, -beta, beta]).reshape([2,2])
            factor = Factor(
                name = ijth_object_name('F', i, j),
                variables = [ith_object_name('V', i), ith_object_name('V', j)],
                log_values = log_values)
            model.add_factor(factor)

    for i in range(nb_vars):
        factor = Factor(
            name = ith_object_name('B', i),
            variables = [ith_object_name('V', i)],
            log_values = np.array([bias[i], -bias[i]]))
        model.add_factor(factor)

    return model

def generate_grid(nb_vars, delta):
    model = GraphicalModel()

    grid_size = int(nb_vars**0.5)
    interaction = delta * np.random.uniform(-1.0, 1.0, 2*grid_size*(grid_size-1))
    bias = np.random.uniform(-0.1, 0.1, [grid_size,grid_size])

    for i in range(grid_size):
        for j in range(grid_size):
            model.add_variable(ijth_object_name('V', i,j))

    edge_set = []
    for x in range(grid_size*grid_size):
        q, m = divmod(x, grid_size)
        if m != grid_size-1:
            edge_set.append([x,x+1])

        if q != grid_size-1:
            edge_set.append([x,x+grid_size])

    for i, e in enumerate(edge_set):
        beta = interaction[i]

        q1, m1 = divmod(e[0], grid_size)
        V1 = ijth_object_name('V', q1,m1)

        q2, m2 = divmod(e[1],grid_size)
        V2 = ijth_object_name('V', q2,m2)

        log_values = np.array([beta, -beta, -beta, beta]).reshape([2,2])
        factor = Factor(name = ith_object_name('F', i),
                        variables = [V1, V2],
                        log_values = log_values)
        model.add_factor(factor)

    for i in range(grid_size):
        for j in range(grid_size):
            log_values = np.array([bias[i,j], -bias[i,j]])
            model.add_factor(Factor(name = ith_object_name('B', i*grid_size + j),
                                 variables = [ijth_object_name('V', i,j)],
                                 log_values = log_values))

    return model

def generate_grid_gmi(m, n, delta):
    '''
        m is number of rows
        n is number of columns
        delta is max strength of interaction
    '''
    model = GraphicalModel()
    G = nx.Graph()
    pos = {}
    nb_vars = m*n
    # grid_size = int(nb_vars**0.5)
    number_of_edges = 2*m*n-m-n
    interaction = delta * np.random.uniform(-1.0, 1.0, number_of_edges)
    bias = np.random.uniform(-0.1, 0.1, [m,n])
    node_colors = ['b']*nb_vars

    # beta = 1 # inverse temperature beta for simulated annealing
    inv_temp = 1 # inverse temperature beta for simulated annealing
    # Z = 1 # partition function

    # inf = int(grid_size/2)+1 # the initially infected node.
    # adding all variables for each node in grid
    # indexed by two coordinates (i,j)
    for i in range(m):
        for j in range(n):
            model.add_variable(ijth_object_name('V', i,j))
            # define the intially infected variables here too
            # if i == inf and j == inf: ...

    # creating all edges
    edge_set = []
    for x in range(nb_vars):
        G.add_node(x)
        # (q,r) are (row,col) coordinates for node x
        q, r = divmod(x, n)

        G.node[x]['x'] = float(q)/m
        G.node[x]['y'] = float(r)/n
        pos[x] = [float(q)/m,float(r)/n]

        if r != n-1:
            edge_set.append([x,x+1])
            G.add_edge(x, x+1)
        if q != m-1:
            edge_set.append([x,x+n])
            G.add_edge(x,x+n)

    # adding all factors
    g = [] # edge weights
    for i, e in enumerate(edge_set):

        g.append(np.random.uniform(0,1))
        # create variable names for edges
        # coordinates for nodes in the grid
        q1, r1 = divmod(e[0], n)
        V1 = ijth_object_name('V', q1,r1)

        q2, r2 = divmod(e[1], n)
        V2 = ijth_object_name('V', q2,r2)

        beta = np.log(1+np.exp(g[i]))/2
        log_values = np.array([beta, -beta, -beta, beta]).reshape([2,2])
        factor = Factor(name = ith_object_name('F', i),
                        variables = [V1, V2],
                        log_values = log_values)
        model.add_factor(factor)


    # randomly select initially infected nodes
    # sigma_in = np.round(np.random.uniform([grid_size,grid_size]))
    # sigma_in = np.zeros([m,n])
    # beta = -(np.log(1-sigma_in[i,j]) - inv_temp)/2
    # assigning

    # assigning buckets to variables

    for i in range(m):
        for j in range(n):
            if i == np.floor(m/2) and j == np.floor(n/2):
                node_colors[i*n+j] = 'r'
                beta = -inv_temp
            else:
                beta = 0
            log_values = np.array([-beta, beta])
            # log_values = np.array([bias[i,j], -bias[i,j]])
            model.add_factor(Factor(name = ith_object_name('B', i*n + j),
                                 variables = [ijth_object_name('V', i,j)],
                                 log_values = log_values))

    # print(enumerate(edge_set))
    # print(nb_vars, model.summary())
    # print(model.factors)
    nx.draw(G,pos, node_color=node_colors)
    print('max degree = {}'.format(np.max([G.degree(node) for node in G.nodes])))
    plt.title('grid size = {}, initially infected node is {}'.format([m,n], [np.floor(i/m), np.floor(j/n)]))
    plt.show()
    return model

def generate_complete_gmi(nb_vars, delta, init_inf, inv_temp):
    model = GraphicalModel()
    node_colors = ['b']*nb_vars
    interaction = delta * np.random.uniform(0, 1.0, nb_vars*(nb_vars-1))

    for i in range(nb_vars):
        model.add_variable(ith_object_name('V', i))

    active = [] # active edges
    for i in range(nb_vars):
        for j in range(i+1, nb_vars):
            beta = interaction[i * nb_vars + j ]
            log_values = np.array([beta, -beta, -beta, beta]).reshape([2,2])
            factor = Factor(
                name = ijth_object_name('F', i, j),
                variables = [ith_object_name('V', i), ith_object_name('V', j)],
                log_values = log_values)
            model.add_factor(factor)

            # IC model
            rand = np.random.uniform(0,1)
            if rand > beta:
                active.append( [i,j] )


    for i in range(nb_vars):
        # define factor definitions
        if i in init_inf:
            node_colors[i] = 'r'
            beta = -inv_temp
        else:
            beta = 0
        log_values = np.array([-beta, beta])
        factor = Factor(
            name = ith_object_name('B', i),
            variables = [ith_object_name('V', i)],
            log_values = log_values)
        model.add_factor(factor)

    model.active = active

    # nx.draw(G, node_color=node_colors)
    # print('max degree = {}'.format(np.max([G.degree(node) for node in G.nodes])))
    # plt.title('complete graph size = {}, initially infected node is {}'.format(nb_vars, 0))
    # plt.show()
    return model

def generate_seattle(G, init_inf):
    model = GraphicalModel()
    # G = nx.Graph()
    N = len(G.nodes)
    inv_temp = 1
    node_colors = ['b']*N

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
        if node in init_inf:
            node_colors[node] = 'r'
            beta = -inv_temp
        else:
            beta = 0
        log_values = np.array([-beta, beta])
        factor = Factor(
            name = ith_object_name('B', node),
            variables = [ith_object_name('V', node)],
            log_values = log_values)
        model.add_factor(factor)
    # print(model.summary())
    return model




UAI_PATH='./graphical_model/UAI/'

def generate_uai(model_name):
    model = GraphicalModel()
    model.name=model_name

    with open(UAI_PATH+model_name+'.uai') as f:
        a = f.readlines()

    content = []
    for c in a:
        content.extend(c.split())

    cnt = 1
    nb_vars = int(content[cnt])
    cardinalities = dict()
    for t in range(nb_vars):
        cnt += 1
        newvar = 'V' + str(t)
        model.add_variable(newvar)
        cardinalities[newvar] = int(content[cnt])

    cnt += 1
    nfactors = int(content[cnt])

    factor_variables = dict()
    for t in range(nfactors):
        cnt += 1
        newfac_name = 'F' + str(t)
        factor_size = int(content[cnt])
        factor_variables[newfac_name] = []
        for t2 in range(factor_size):
            cnt += 1
            factor_variables[newfac_name].append('V' + str(content[cnt]))

    for t in range(nfactors):
        cnt += 1
        value_num = int(content[cnt])
        newfac_name = 'F' + str(t)
        values = []
        for vt2 in range(value_num):
            cnt += 1
            values.append(float(content[cnt]))

        values = np.reshape(
            values,
            [cardinalities[var] for var in factor_variables[newfac_name]])
        factor = Factor(
            name = newfac_name,
            variables = factor_variables[newfac_name],
            values = values)
        model.add_factor(factor)

    return model

'''
def generate_model(model_type = 'forney_complete', **kwargs):
    if 'seed' in kwargs:
        np.random.seed(kwargs['seed'])

    model = GraphicalModel()
    if model_type == 'forney_complete':
        nfactors = kwargs['nfactors']
        nstates = kwargs['nstates']
        delta = kwargs['delta']

        factorNvariables = {i : [] for i in range(nfactors)}
        for i in range(nfactors):
            for j in range(i+1, nfactors):
                variable = ijth_object_name('V', i,j)
                model.add_variable(variable)
                factorNvariables[i].append(variable)
                factorNvariables[j].append(variable)
        for i in range(nfactors):
            cardinality = [nstates for j in range(nfactors - 1)]
            factor_log_value = delta * np.random.normal(0.0, 1.0, cardinality)
            model.add_factor(Factor(name = ith_object_name('F', i),
                                 variables = factorNvariables[i],
                                 log_values = factor_log_value))

        true_logZ = BucketElimination(model).run()

    if model_type == 'forney_grid':
        nfactors = kwargs['nfactors']
        nstates = kwargs['nstates']
        delta = kwargs['delta']

        grid_size = nfactors**0.5
        factorNvariables = {i : [] for i in range(nfactors)}
        for i in range(nfactors):
            q, r = divmod(i, grid_size)
            if r != grid_size-1:
                variable = ijth_object_name('V', i, i + 1)
                model.add_variable(variable)
                factorNvariables[i].append(variable)
                factorNvariables[i + 1].append(variable)
            if q != grid_size-1:
                variable = ijth_object_name('V', i, i + grid_size)
                model.add_variable(variable)
                factorNvariables[i].append(variable)
                factorNvariables[i + grid_size].append(variable)

        for i in range(nfactors):
            cardinality = [nstates for j in range(len(factorNvariables[i]))]
            factor_log_value = delta * np.random.normal(0.0, 1.0, cardinality)
            model.add_factor(Factor(name = ith_object_name('F', i),
                                 variables = factorNvariables[i],
                                 log_values = factor_log_value))

        true_logZ = BucketElimination(model).run()

    if model_type == 'complete':
        nb_vars = kwargs['nb_vars']
        delta = kwargs['delta']

        interaction = delta * np.random.uniform(-1.0, 1.0, nb_vars * (nb_vars - 1))
        bias = np.random.uniform(-0.1, 0.1, [nb_vars])

        for i in range(nb_vars):
            model.add_variable(ith_object_name('V', i))

        for i in range(nb_vars):
            for j in range(i+1, nb_vars):
                beta = interaction[i * nb_vars + j ]
                factor_log_values = np.array([beta, -beta, -beta, beta]).reshape([2,2])
                factor = Factor(name = ijth_object_name('F', i, j),
                                variables = [ith_object_name('V', i), ith_object_name('V', j)],
                                log_values = factor_log_values)
                model.add_factor(factor)

        for i in range(nb_vars):
            bias_log_values = np.array([bias[i], -bias[i]])
            model.add_factor(Factor(name = ith_object_name('B', i),
                                variables = [ith_object_name('V', i)],
                                log_values = bias_log_values))

        true_logZ = BucketElimination(model).run()

    elif model_type == 'grid':
        nb_vars = kwargs['nb_vars']
        delta = kwargs['delta']
        interaction_type = kwargs['interaction_type']

        grid_size = int(nb_vars**0.5)

        if interaction_type == 'random':
            #interaction = delta * np.random.normal(0.0, 1.0, 2*grid_size*(grid_size-1))
            interaction = delta * np.random.uniform(-1.0, 1.0, 2*grid_size*(grid_size-1))
            #interaction = delta * np.random.uniform(0.0, 1.0, 2*grid_size*(grid_size-1))

        if interaction_type == 'uniform':
            interaction = delta * np.ones(2*grid_size*(grid_size-1))

        #bias = np.random.normal(0.0, 0.01, [grid_size,grid_size])
        bias = np.random.uniform(-0.1, 0.1, [grid_size,grid_size])
        #bias = np.random.uniform(0.0, 0.0, [grid_size,grid_size])

        for i in range(grid_size):
            for j in range(grid_size):
                model.add_variable(ijth_object_name('V', i,j))

        edge_set = []
        for x in range(grid_size*grid_size):
            q, m = divmod(x, grid_size)
            if m != grid_size-1:
                edge_set.append([x,x+1])

            if q != grid_size-1:
                edge_set.append([x,x+grid_size])

        for i, e in enumerate(edge_set):
            beta = interaction[i]

            q1, m1 = divmod(e[0], grid_size)
            V1 = ijth_object_name('V', q1,m1)

            q2, m2 = divmod(e[1],grid_size)
            V2 = ijth_object_name('V', q2,m2)

            factor_log_values = np.array([beta, -beta, -beta, beta]).reshape([2,2])
            factor = Factor(name = ith_object_name('F', i),
                            variables = [V1, V2],
                            log_values = factor_log_values)
            model.add_factor(factor)

        for i in range(grid_size):
            for j in range(grid_size):
                bias_log_values = np.array([bias[i,j], -bias[i,j]])
                model.add_factor(Factor(name = ith_object_name('B', i*grid_size + j),
                                     variables = [ijth_object_name('V', i,j)],
                                     log_values = bias_log_values))

        model.grid_size = grid_size
        model.interaction = interaction

        true_logZ = BucketElimination(model).run(elimination_order_method='not_random')

    if model_type == 'uai':
        filename = kwargs['filename']

        with open(file_dir_path+'/UAI/' + filename) as f:
            a = f.readlines()

        content = []
        for c in a:
            content.extend(c.split())

        cnt = 1
        nb_vars = int(content[cnt])
        cardinalities = dict()
        for t in range(nb_vars):
            cnt += 1
            newvar = 'V' + str(t)
            model.add_variable(newvar)
            cardinalities[newvar] = int(content[cnt])

        cnt += 1
        nfactors = int(content[cnt])

        factor_variables = dict()
        for t in range(nfactors):
            cnt += 1
            newfac_name = 'F' + str(t)
            factor_size = int(content[cnt])
            factor_variables[newfac_name] = []
            for t2 in range(factor_size):
                cnt += 1
                factor_variables[newfac_name].append('V' + str(content[cnt]))

        for t in range(nfactors):
            cnt += 1
            value_num = int(content[cnt])
            newfac_name = 'F' + str(t)
            values = []
            for vt2 in range(value_num):
                cnt += 1
                values.append(float(content[cnt]))

            model.add_factor(Factor(name = newfac_name,
                                 variables = factor_variables[newfac_name],
                                 values = np.reshape(values, [cardinalities[var] for var in factor_variables[newfac_name]])))



        content = [c.strip() for c in a]
        nb_vars = int(content[1])
        cardinalities = [int(c) for c in content[2].split()]
        for i in range(nb_vars):
            model.add_variable('V' + str(i))

        nfactors = int(content[3])
        for i in range(nfactors):
            line = [int(c) for c in content[4+i].split()]
            name = 'F' + str(i)
            variables = ['V' + str(j) for j in line[1:]]
            card = [cardinalities[j] for j in line[1:]]
            model.add_factor(Factor(name = name,
                                 variables = variables,
                                 log_values = np.zeros(card)))

        cnt = 6 + nfactors

        for i in range(nfactors):
            values = []
            while len(values) != np.prod(model.factors[i].cardinality):
                values = values + [float(c) for c in content[cnt].split()]
                cnt += 1

            model.factors[i].values = np.reshape(values, model.factors[i].cardinality)

            if filename == 'Pedigree_11.uai':
                cnt += 1
            else:
                cnt += 2

        with open(file_dir_path + '/UAI/' + filename+'.PR') as f1:
            a1 = f1.readlines()
        content1 = [c1.strip() for c1 in a1]
        true_logZ = float(content1[1])*np.log(10)

    return model, true_logZ
'''
