
from tools import *
from mean_field import MeanField
from belief_propagation import BeliefPropagation

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
    quit()

def testing_run_time():
    init_inf = [0]
    runtime = []
    TAUS = [60, 70, 80, 90, 100, 110, 120]
    for H_a in [1.0]:
        for MU in [0.00138985]:
            for TAU in TAUS:
                G = extract_seattle_data(TAU, MU)
                seattle = generate_seattle(G, init_inf, H_a)
                # =====================================
                # Compute partition function for Seattle GM
                # =====================================
                t1 = time.time()
                Z = BucketRenormalization(seattle, ibound=10).run(max_iter=1)
                t2 = time.time()
                # =====================================
                print('partition function = {}'.format(Z))
                print('time taken for GBR = {}'.format(t2-t1))
                runtime.append(t2-t1)
    plt.plot(TAUS, runtime)
    plt.ylabel("GBR Runtime")
    plt.xlabel(R"minium threshold $\tau$")
    plt.show()
    quit()

def testing_partition_function_dependence_on_TAU():
    init_inf = [0]
    pf = []
    TAUS = [70, 80, 90, 100, 110, 120]
    for H_a in [1.0]:
        for MU in [0.00138985]:
            for TAU in TAUS:
                G = extract_seattle_data(TAU, MU)
                seattle = generate_seattle(G, init_inf, H_a)
                # =====================================
                # Compute partition function for Seattle GM
                # =====================================
                t1 = time.time()
                Z = BucketRenormalization(seattle, ibound=10).run(max_iter=1)
                t2 = time.time()
                # =====================================
                print('partition function = {}'.format(Z))
                print('time taken for GBR = {}'.format(t2-t1))
                pf.append(Z)
    plt.plot(TAUS, pf)
    plt.ylabel("Z Value")
    plt.xlabel(R"minium threshold $\tau$")
    pls.savefig("Z_dependence_on_tau.png")
    plt.show()
    quit()

def create_little_GM():
    case = 'seattle'
    G = extract_data(case, TAU=120, MU=0.02)

    # infect Tract 0 and compute
    init_inf = [0]
    H_a = 0.1
    model = generate_graphical_model(case, G, init_inf, H_a, condition_on_init_inf=False)

    V52 = ith_object_name('V', 52)
    # choose node 52 and its neighbors
    adjacent_factors = model.get_adj_factors(V52)
    # collect unique variable names from neighbors of V52
    variables = list(set([var for fac in adjacent_factors for var in fac.variables ]))

    tiny_model = GraphicalModel()
    for fac in adjacent_factors:
        tiny_model.add_variables_from(variables)
        tiny_model.add_factor(fac)
    tiny_model.summary()

    # now condition on infected node
    update_MF_of_neighbors_of(tiny_model, V52)

    # @time_it
    # BE_Z = BucketElimination(tiny_model).run()
    #
    # @time_it
    # GBR_Z = BucketRenormalization(tiny_model).run()
    #
    # @time_it
    # BP_Z = BeliefPropagation(tiny_model).run()
    #
    # @time_it
    # MF_Z = MeanField(tiny_model).run()
    # now compute marginals using BE, GBR, BP, and MF

def implement_BP(case = 'seattle', init_inf = [0], H_a = 0.1, MU = 0.002, TAU = -1):

    G = extract_data(case, TAU, MU)

    # infect Tract 0 and compute

    model = generate_graphical_model(case, G, init_inf, H_a)
    degrees = [model.degree(var) for var in model.variables]

    BP = BeliefPropagation(model).run()

    N = len(model.variables)
    filename = "BP_"+case+"_marg_prob_init_inf={}_H_a={}_MU={}_TAU={}.csv".format(init_inf, H_a, MU, TAU)
    utils.append_to_csv(filename, ['Tract index', 'P'])
    for index in range(N+1):
        if index not in init_inf:
            # print('P( x_{} = {} ) = {}'.format(index, 1, ))
            utils.append_to_csv(filename, [index, BP['marginals']['MARGINAL_V{}'.format(index)]] )
    utils.append_to_csv(filename, ['whole GM',BP['logZ']])

def compare_BP_and_GBR():
    case = 'seattle'
    MU = 0.002
    TAU = 120
    H_a = 0.1
    init_inf = [0]
    filename = "BP_"+case+"_marg_prob_init_inf={}_H_a={}_MU={}_TAU={}.csv".format(init_inf, H_a, MU, TAU)
    BP_data = utils.read_csv(filename)
    filename = case+"_marg_prob_init_inf={}_H_a={}_MU={}_TAU={}.csv".format(init_inf, H_a, MU, float(TAU))
    GBR_data = utils.read_csv(filename)
    print(BP_data)
    print(GBR_data)
    N = len(BP_data)-1
    error = []
    for line in range(N):
        if line == 0: continue
        print("line number {}".format(line))
        print("marginal using BP = {}\t marginal using GBR = {}".format(float(BP_data[line][1]),float(GBR_data[line][-1])))
        err = float(BP_data[line][1])-float(GBR_data[line][-1])
        print("error = {}".format(float(BP_data[line][1])-float(GBR_data[line][-1])))
        error.append(err)

    plt.plot(range(len(error)), np.log(np.abs(error)))
    plt.xlabel('node number')
    plt.ylabel('log of absolute difference in probability')
    plt.title('Comparing Marginals of BP and GBR\n for init_inf[0], H_a = 0.1, TAU = 120, MU = 0.002')
    plt.show()
    print(np.max(np.abs(error)))

# for tau in [100-i*20 for i in range(4)]:
#     print(R"$\tau$ = {}".format(tau))
#     implement_BP(case = 'seattle', init_inf = [0], H_a = 0.1, MU = 0.002, TAU = tau)
# implement_BP(case = 'seattle', init_inf = [0], H_a = 0.1, MU = 0.002, TAU = 0)
# quit()
# decrease J to 0
print("exp 1")
implement_BP(case = 'seattle', init_inf = [0], H_a = 0.1, MU = 0.001, TAU = 0)
implement_BP(case = 'seattle', init_inf = [0], H_a = 0.1, MU = 0.0001, TAU = 0)
implement_BP(case = 'seattle', init_inf = [0], H_a = 0.1, MU = 0.00001, TAU = 0)
# implement_BP(case = 'seattle', init_inf = [0], H_a = 0.1, MU = 0.000001, TAU = -1)
print("exp 2")
# increase H_a to infinity
implement_BP(case = 'seattle', init_inf = [0], H_a = 1.0, MU = 0.002, TAU = 0)
implement_BP(case = 'seattle', init_inf = [0], H_a = 10.0, MU = 0.002, TAU = 0)
print("exp 3")
# increase J to infinity
implement_BP(case = 'seattle', init_inf = [0], H_a = 0.1, MU = 0.01, TAU = 0)
implement_BP(case = 'seattle', init_inf = [0], H_a = 0.1, MU = 0.1, TAU = 0)
implement_BP(case = 'seattle', init_inf = [0], H_a = 0.1, MU = 0.5, TAU = 0)
implement_BP(case = 'seattle', init_inf = [0], H_a = 0.1, MU = 0.9, TAU = 0)
# compare_BP_and_GBR()
