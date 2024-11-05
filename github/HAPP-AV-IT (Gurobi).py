# ==================================================================================================================== #
# |||      * Household Activity Pattern Problem with Automated Vehicle-enabled Intermodal Trips (HAPP-AV-IT) *     ||| #
# |||                                                                                                              ||| #
# |||                                          Developed by Younghun Bahk                                          ||| #
# |||                                   Version 3.62p / Last Update: 2024.11.05.                                   ||| #
# ==================================================================================================================== #

import time
import pandas as pd
import networkx as nx
import pickle
from collections import OrderedDict
import gurobipy as gp
from gurobipy import GRB

start_time = time.time()

# ==================================================================================================================== #
# MODEL SETTINGS
# ==================================================================================================================== #
print("* MODEL SETTINGS")

# Scenario settings
hh_sce = 0  # 0: virtual households, 1: SANDAG default data, 2: SANDAG reduced vehicle ownerships
nw_sce = 1  # 1: 2 transit lines, 2: 5 transit lines

# Network initialization switch
nw_initialize = "off"  # turn on when running for the first time

# Stopping criteria switch
timelimit = "on"  # stop when maximum runtime reached
timelimit_v = 1800
mipgap = "off"  # stop when gap tolerance reached
mipgap_v = 0.1
incumbent = "on"  # stop when maximum incumbent time reached
incumbent_v = 900
first_solution = "off"  # stop when first feasible solution found

print("Household set:", hh_sce, "/ Network scenario:", nw_sce)
print("Network initialization:", nw_initialize)
print("Stopping rule 1 (time limit):   ", timelimit_v, "seconds / switch", timelimit)
print("Stopping rule 2 (gap tolerance):", str(mipgap_v * 100) + str("% / switch"), mipgap)
print("Stopping rule 3 (max incumbent):", incumbent_v, "seconds / switch", incumbent)
print("Stopping rule 4 (1st solution):  switch", first_solution)
print("")

# ==================================================================================================================== #
# NETWORK INPUT
# ==================================================================================================================== #
print("* NETWORK INPUT")

# Network generation
node = pd.read_excel(r'input\network_' + str(nw_sce) + '.xlsx', sheet_name='node')
link_rd = pd.read_excel(r'input\network_' + str(nw_sce) + '.xlsx', sheet_name='link')
link_tr = pd.read_excel(r'input\network_' + str(nw_sce) + '.xlsx', sheet_name='link_trn')

location = {node['id'].iloc[i]: (node['x'].iloc[i], node['y'].iloc[i]) for i in range(len(node))}

print("Generating road network...")
nw_road = nx.DiGraph()  # road network
nw_road.add_nodes_from(location.keys())
for n, p in location.items():
    nw_road.nodes[n]['location'] = p
for i in range(len(link_rd)):
    nw_road.add_edge(link_rd['from'].iloc[i], link_rd['to'].iloc[i], travel_time=link_rd['time_avg'].iloc[i],
                     length=link_rd['length'].iloc[i], type=link_rd['type'].iloc[i])  # fixed link travel time

print("Generating rail network...")
nw_rail = nx.DiGraph()  # transit network
nw_rail.add_nodes_from(location.keys())
for n, p in location.items():
    nw_rail.nodes[n]['location'] = p
for i in range(len(link_tr)):
    nw_rail.add_edge(link_tr['from'].iloc[i], link_tr['to'].iloc[i], travel_time=link_tr['time'].iloc[i],
                     length=link_tr['length'].iloc[i], type=link_tr['type'].iloc[i])
print("")

# ==================================================================================================================== #
# ACCESSIBILITY
# ==================================================================================================================== #
print("* ACCESSIBILITY")

# Shortest path travel times and distances
print("Listing nodes by type...")
ls_hl = []  # list of Home Location nodes in the network
ls_al = []  # list of Activity Location nodes in the network
ls_ts = []  # list of Transit Station nodes in the network
ls_ps = []  # list of Parking Space nodes in the network
pks_fee = OrderedDict()  # dictionary of parking fees
for i in range(len(node)):
    if node['type'].iloc[i] in ['HL']:
        ls_hl.append(node['id'].iloc[i])
    if node['type'].iloc[i] in ['AL']:
        ls_al.append(node['id'].iloc[i])
    if node['type'].iloc[i] in ['TS']:
        ls_ts.append(node['id'].iloc[i])
    if node['type'].iloc[i] in ['PS', 'HL']:
        if node['type'].iloc[i] in ['PS']:
            ls_ps.append(node['id'].iloc[i])
        pks_fee[node['id'].iloc[i]] = node['fee'].iloc[i]
if nw_sce == 2:
    ls_hub = [305, 306, 308, 312, 313, 315, 321, 323, 327, 329, 331, 336, 337, 338]
else:
    ls_hub = [306, 313]  # transit transfer stations

if nw_initialize == "on":
    print("Calculating shortest paths on road network...")
    tt = OrderedDict()  # dictionary of shortest travel times
    td = OrderedDict()  # dictionary of shortest travel distances
    count = 0
    for i in ls_hl + ls_al + ls_ts + ls_ps:
        tt[i] = OrderedDict()
        td[i] = OrderedDict()
        for j in ls_hl + ls_al + ls_ts + ls_ps:
            tt[i][j] = nx.shortest_path_length(nw_road, i, j, weight='travel_time')
            td[i][j] = nx.shortest_path_length(nw_road, i, j, weight='length')
        count += 1
        if count % 10 == 0:
            print("-", count, "/", len(ls_hl + ls_al + ls_ts + ls_ps), "nodes completed...")
    print("-", len(ls_hl + ls_al + ls_ts + ls_ps), "/", len(ls_hl + ls_al + ls_ts + ls_ps), "nodes completed...")
    with open(r'input\tt_' + str(nw_sce) + '.pickle', "wb") as f:
        pickle.dump(tt, f)
    with open(r'input\td_' + str(nw_sce) + '.pickle', "wb") as f:
        pickle.dump(td, f)
    print("")

print("Reading saved travel times and distances...")
with open(r'input\tt_' + str(nw_sce) + '.pickle', "rb") as f:
    tt = pickle.load(f)
with open(r'input\td_' + str(nw_sce) + '.pickle', "rb") as f:
    td = pickle.load(f)

# Location search for the nearest transit station, transit hub, and parking space
print("Sorting transit stations and parking spaces...")
stn_access = OrderedDict()  # dictionary of the shortest travel times from AL/HL to TS
hub_access = OrderedDict()  # dictionary of the shortest travel times from HL to TS_hub
pks_access_short = OrderedDict()  # dictionary of the shortest travel times from AL/HL/TS to PS
pks_access_cheap = OrderedDict()  # dictionary of the parking fees
stn_sorted = OrderedDict()  # TS list sorted by travel time
hub_sorted = OrderedDict()  # TS_hub list sorted by travel time
pks_sorted_short = OrderedDict()  # PS list sorted by travel time
pks_sorted_cheap = OrderedDict()  # PS list sorted by parking fee and travel time
for i in ls_hl + ls_al:
    stn_access[i] = OrderedDict()
    for j in ls_ts:
        stn_access[i][j] = tt[i][j]
    stn_sorted[i] = list(OrderedDict(sorted(stn_access[i].items(), key=lambda item: item[1])).keys())
for i in ls_hl:
    hub_access[i] = OrderedDict()
    for j in ls_hub:
        hub_access[i][j] = tt[i][j]
    hub_sorted[i] = list(OrderedDict(sorted(hub_access[i].items(), key=lambda item: item[1])).keys())
for i in ls_hl + ls_al + ls_ts:
    pks_access_short[i] = OrderedDict()
    pks_access_cheap[i] = OrderedDict()
    for j in ls_ps:
        pks_access_short[i][j] = tt[i][j]
    pks_sorted_short[i] = list(OrderedDict(sorted(pks_access_short[i].items(), key=lambda item: item[1])).keys())
    for k in pks_sorted_short[i]:
        pks_access_cheap[i][k] = pks_fee[k]
    pks_sorted_cheap[i] = list(OrderedDict(sorted(pks_access_cheap[i].items(), key=lambda item: item[1])).keys())

# Transit walk access/egress time input
print("Calculating transit station access/egress walk times and transit travel times...")
tt_wlk = OrderedDict()
tt_trn = OrderedDict()  # dictionary of shortest transit travel times
for i in ls_hl + ls_al + ls_ts:
    tt_wlk[i] = OrderedDict()
    tt_trn[i] = OrderedDict()
    for j in ls_hl + ls_al + ls_ts:
        if i in ls_hl + ls_al and j in ls_hl + ls_al:  # 80 m/min for walk speed, 1 min for walk time in the station
            tt_wlk[i][j] = td[i][stn_sorted[i][0]] / 80 + 1 + td[stn_sorted[j][0]][j] / 80
            tt_trn[i][j] = nx.shortest_path_length(nw_rail, stn_sorted[i][0], stn_sorted[j][0], weight='travel_time')
        if i in ls_ts and j in ls_hl + ls_al:
            tt_wlk[i][j] = td[stn_sorted[j][0]][j] / 80 + 1
            tt_trn[i][j] = nx.shortest_path_length(nw_rail, i, stn_sorted[j][0], weight='travel_time')
        if i in ls_hl + ls_al and j in ls_ts:
            tt_wlk[i][j] = td[i][stn_sorted[i][0]] / 80 + 1
            tt_trn[i][j] = nx.shortest_path_length(nw_rail, stn_sorted[i][0], j, weight='travel_time')
        if i in ls_ts and j in ls_ts:
            tt_wlk[i][j] = 1
            tt_trn[i][j] = nx.shortest_path_length(nw_rail, i, j, weight='travel_time')
print("")

# ==================================================================================================================== #
# HOUSEHOLD ACTIVITY INPUT
# ==================================================================================================================== #
print("* HOUSEHOLD ACTIVITY INPUT")

timer = OrderedDict()
sce_result = []

# Household activity profile input
hh_list = pd.read_excel(r'input\household_' + str(hh_sce) + '.xlsx', sheet_name='household')
ac_list = pd.read_excel(r'input\household_' + str(hh_sce) + '.xlsx', sheet_name='activity')

# Household activity travel output
sce_column = ['hhid', 'hh_size', 'activities', 'PAVs', 'PAV_trips', 'SAV_trips', 'trn_trips', 'im_trips', 'PAV_pooling',
              'PAV_time', 'ob_PAV_time', 'dh_PAV_time', 'SAV_time', 'trn_time', 'prk_time', 'wlk_time', 'prs_time',
              'PAV_dist', 'ob_PAV_dist', 'dh_PAV_dist', 'SAV_dist', 'VKT', 'out_HL_prk', 'prk_fee', 'SAV_fare',
              'trn_fare', 'runtime', 'obj_value', 'gap', 'sol_dur', 'optimized?']
df_sce_result = pd.DataFrame(columns=sce_column)
df_sce_result.to_csv(r'result_sce' + str(hh_sce) + str(nw_sce) + '.csv', index=False)

# Input start
for h in range(len(hh_list)):
    hh_result = []
    hhid = hh_list['hh_id'].iloc[h]
    print("Household", hhid, "travel planning started.")

    # Timer
    timer[h] = time.time()

    # Physical locations of HL and AL for each household
    HL = hh_list['home'].iloc[h]  # physical home location
    AL = OrderedDict()  # physical activity locations
    for i in range(len(ac_list)):
        if ac_list['hh_id'].iloc[i] == hhid:
            aid = ac_list['a_id'].iloc[i]
            AL[aid] = ac_list['nw_node'].iloc[i]

    # Activity node input
    print("Assigning activity nodes...")
    A = len(AL)  # number of household activities
    N_0 = [0]  # initial node
    N_hp = []  # HL pickup nodes
    N_ad = []  # AL drop-off nodes
    N_td_1 = []  # TS drop-off nodes nearest from HL
    N_td_2 = []  # TS drop-off nodes at transit hub
    N_pk_0s = []  # PS nearest from AL
    N_pk_1s = []  # PS nearest from TS pickup 1
    N_pk_2s = []  # PS nearest from transit hub
    N_pk_0c = []  # PS nearest from AL among cheapest PS
    N_pk_1c = []  # PS nearest from TS pickup 1 among cheapest PS
    N_pk_2c = []  # PS nearest from transit hub among cheapest PS
    N_pk_h = []  # HL (parking at home)
    N_ap = []  # AL pickup nodes
    N_tp_1 = []  # TS pickup nodes nearest from HL
    N_tp_2 = []  # TS pickup nodes at transit hub
    N_hd = []  # HL drop-off nodes
    N_hs = []  # HL stay-at-home nodes
    N_f = [16 * A + 1]  # final node
    for i in range(A):
        N_hp.append(i + 1)
        N_ad.append(i + 1 * A + 1)
        N_td_1.append(i + 2 * A + 1)
        N_td_2.append(i + 3 * A + 1)
        N_pk_0s.append(i + 4 * A + 1)
        N_pk_1s.append(i + 5 * A + 1)
        N_pk_2s.append(i + 6 * A + 1)
        N_pk_0c.append(i + 7 * A + 1)
        N_pk_1c.append(i + 8 * A + 1)
        N_pk_2c.append(i + 9 * A + 1)
        N_pk_h.append(i + 10 * A + 1)
        N_ap.append(i + 11 * A + 1)
        N_tp_1.append(i + 12 * A + 1)
        N_tp_2.append(i + 13 * A + 1)
        N_hd.append(i + 14 * A + 1)
        N_hs.append(i + 15 * A + 1)
    N_td = N_td_1 + N_td_2  # all candidate TS drop-off nodes
    N_pk = N_pk_0s + N_pk_1s + N_pk_2s + N_pk_0c + N_pk_1c + N_pk_2c + N_pk_h  # candidate PS nodes
    N_tp = N_tp_1 + N_tp_2  # all candidate TS pickup nodes
    N_PU = N_hp + N_ap + N_tp
    N_DO = N_ad + N_td + N_hd
    N_PUDO = N_PU + N_DO
    N_hat = N_hp + N_ad + N_td + N_pk + N_ap + N_tp + N_hd + N_hs  # all activity nodes but initial and final nodes
    N = N_0 + N_hat + N_f  # all activity nodes

    # Physical locations of each activity node
    print("Assigning corresponding physical locations...")
    L = OrderedDict()  # physical location of each activity node
    for i in range(A):
        L[N_0[0]] = L[N_f[0]] = L[N_hp[i]] = L[N_hd[i]] = L[N_pk_h[i]] = L[N_hs[i]] = HL
        L[N_ad[i]] = L[N_ap[i]] = AL[i + 1]
        L[N_td_1[i]] = L[N_tp_1[i]] = stn_sorted[HL][0]
        L[N_td_2[i]] = L[N_tp_2[i]] = hub_sorted[HL][0]
        L[N_pk_0s[i]] = pks_sorted_short[AL[i + 1]][0]
        L[N_pk_1s[i]] = pks_sorted_short[L[N_tp_1[i]]][0]
        L[N_pk_2s[i]] = pks_sorted_short[L[N_tp_2[i]]][0]
        L[N_pk_0c[i]] = pks_sorted_cheap[AL[i + 1]][0]
        L[N_pk_1c[i]] = pks_sorted_cheap[L[N_tp_1[i]]][0]
        L[N_pk_2c[i]] = pks_sorted_cheap[L[N_tp_2[i]]][0]

    # Travel times and distances input
    print("Generating travel times and distances for each household trip...")
    t = OrderedDict()  # road network vehicle travel time
    d = OrderedDict()  # road network travel distance
    t_wlk = OrderedDict()  # road network walk time
    t_trn = OrderedDict()  # rail network vehicle travel time
    for i in N:
        t[i] = OrderedDict()
        d[i] = OrderedDict()
        if i in N_PUDO:
            t_wlk[i] = OrderedDict()
            t_trn[i] = OrderedDict()
        for j in N:
            t[i][j] = tt[L[i]][L[j]]
            d[i][j] = td[L[i]][L[j]]
            if i in N_PUDO and j in N_PUDO:
                t_wlk[i][j] = tt_wlk[L[i]][L[j]]
                t_trn[i][j] = tt_trn[L[i]][L[j]]

    # Time windows input
    print("Generating household activity time windows...")
    a = OrderedDict()  # time window: earliest arrival time
    b = OrderedDict()  # time window: latest arrival time
    s = OrderedDict()  # activity duration
    a_0 = hh_list['earliest'].iloc[h]  # time window: earliest activity starting time for each household
    b_f = hh_list['latest'].iloc[h]  # time window: latest home return time for each household

    for i in range(len(ac_list)):
        if ac_list['hh_id'].iloc[i] == hhid:
            aid = ac_list['a_id'].iloc[i]
            a[aid + A] = ac_list['do_early'].iloc[i]
            b[aid + A] = ac_list['do_late'].iloc[i]
            a[aid + 11 * A] = ac_list['pu_early'].iloc[i]
            b[aid + 11 * A] = ac_list['pu_late'].iloc[i]
            for j in range(15):
                if j == 1:
                    s[aid + A] = ac_list['duration'].iloc[i]
                else:
                    s[aid + j * A] = 0

    # Parking fees input
    print("Generating parking fees for each parking choice...")
    pkf = OrderedDict()
    for i in N_pk:
        pkf[i] = pks_fee[L[i]]

    # Person (household member) and vehicle input
    print("Generating household members and vehicles...")
    P = []
    V = []
    for i in range(hh_list['size_p'].iloc[h]):
        P.append(i + 1)
    for i in range(hh_list['size_v'].iloc[h]):
        V.append(i + 1)

    # Vehicle/person restrictions on activities
    print("Generating vehicle/person restrictions on activities...")
    theta = OrderedDict()
    for p in P:
        theta[p] = []
        for j in range(len(ac_list)):
            if ac_list['hh_id'].iloc[j] == hhid and ac_list['traveler'].iloc[j] == p:
                theta[p].append(ac_list['a_id'].iloc[j])
    print("")

# ==================================================================================================================== #
# HOUSEHOLD TRAVEL PLANNING MODEL
# ==================================================================================================================== #
    print("* HOUSEHOLD AV TRAVEL PLANNING MODEL")

    # Stopping rule 3: maximum incumbent time
    def cb(model, where):
        if where == GRB.Callback.MIPNODE:
            obj = model.cbGet(GRB.Callback.MIPNODE_OBJBST)
            if abs(obj - model._cur_obj) > 1e-8:
                model._cur_obj = obj
                model._time = time.time()
        if time.time() - model._time > incumbent_v and model._cur_obj != float('inf'):
            model.terminate()

    mdl = gp.Model("HAPP-AV-IT")

    # Decision variables ============================================================================================= #
    X = {}  # PAV trip
    Y = {}  # SAV trip
    Z = {}  # transit trip
    H = {}  # person trip
    T = {}  # arrival time
    T_0 = {}  # initial home departure time
    T_f = {}  # final home return time
    Q = {}  # wait time
    K = {}  # parking duration time
    rho = {}  # onboard activities when leaving a node
    for i in N:
        X[i] = {}
        Y[i] = {}
        Z[i] = {}
        H[i] = {}
        T[i] = mdl.addVar(lb=180, ub=1620, vtype=GRB.CONTINUOUS, name="Arrival time")
        Q[i] = mdl.addVar(lb=0, ub=30, vtype=GRB.CONTINUOUS, name="Waiting time")
        rho[i] = {}
        for j in N:
            X[i][j] = {}
            Y[i][j] = mdl.addVar(lb=0, ub=1, vtype=GRB.BINARY, name="SAV trip")
            Z[i][j] = mdl.addVar(lb=0, ub=1, vtype=GRB.BINARY, name="Transit trip")
            H[i][j] = {}
            for v in V:
                X[i][j][v] = mdl.addVar(lb=0, ub=1, vtype=GRB.BINARY, name="PAV trip")
            for p in P:
                H[i][j][p] = mdl.addVar(lb=0, ub=1, vtype=GRB.BINARY, name="Household member trip")
        for v in V:
            rho[i][v] = mdl.addVar(lb=0, ub=4, vtype=GRB.INTEGER, name="Onboard passengers")
    for v in V:
        T_0[v] = mdl.addVar(lb=180, ub=1620, vtype=GRB.CONTINUOUS, name="Initial node time, PAV")
        T_f[v] = mdl.addVar(lb=180, ub=1620, vtype=GRB.CONTINUOUS, name="Final node time, PAV")
    for p in P:
        T_0[p] = mdl.addVar(lb=180, ub=1620, vtype=GRB.CONTINUOUS, name="Initial node time, person")
        T_f[p] = mdl.addVar(lb=180, ub=1620, vtype=GRB.CONTINUOUS, name="Final node time, person")
    for i in N_pk:
        K[i] = mdl.addVar(lb=0, ub=1440, vtype=GRB.CONTINUOUS, name="Parking duration time")

    # Parameters ===================================================================================================== #
    beta_PAVwt = 0.16  # USD per minute
    beta_PAViv = 0.08  # USD per minute
    beta_PAVop = 0.0003  # USD per meter (from DOT)
    beta_SAVwt = 0.16  # USD per minute
    beta_SAViv = 0.13  # USD per minute
    beta_SAVfb = 1.00  # USD per trip
    beta_SAVfd = 0.0015  # USD per meter
    beta_TRNwk = 0.22  # USD per minute
    beta_TRNwt = 0.12  # USD per minute
    beta_TRNiv = 0.08  # USD per minute
    beta_TRNfr = 1.75  # USD per trip
    w_SAV = 6
    w_TRN = 3
    M = 9999  # a large number

    # Objective function ============================================================================================= #
    # Eqn 1
    HH_cost = \
        beta_PAVwt * gp.quicksum(Q[i] for i in N_ap + N_tp) + \
        beta_PAViv * gp.quicksum(t[i][j] * H[i][j][p] for i in N for j in N for p in P) + \
        beta_PAVop * gp.quicksum(d[i][j] * X[i][j][v] for i in N for j in N for v in V) + \
        gp.quicksum(pkf[i] * K[i] for i in N_pk) + \
        beta_SAVwt * gp.quicksum(w_SAV * Y[i][j] for i in N for j in N) + \
        (beta_SAViv - beta_PAViv) * gp.quicksum(t[i][j] * Y[i][j] for i in N for j in N) + \
        beta_SAVfb * gp.quicksum(Y[i][j] for i in N for j in N) + \
        beta_SAVfd * gp.quicksum(d[i][j] * Y[i][j] for i in N for j in N) + \
        beta_TRNwk * gp.quicksum(t_wlk[i][j] * Z[i][j] for i in N_PUDO for j in N_PUDO) + \
        beta_TRNwt * gp.quicksum(w_TRN * Z[i][j] for i in N_PUDO for j in N_PUDO) + \
        beta_TRNiv * gp.quicksum((t_trn[i][j] - t[i][j]) * Z[i][j] for i in N_PUDO for j in N_PUDO) + \
        beta_TRNfr * gp.quicksum(Z[i][j] for i in N_PUDO for j in N_PUDO)
    mdl.setObjective(HH_cost, GRB.MINIMIZE)

    # Constraint set 1: Spatial connectivity constraints on vehicles ================================================= #
    # mdl.addConstr(gp.quicksum(Y[i][j] for i in N_PUDO for j in N_PUDO) == 0, name="No SAV")
    # mdl.addConstr(gp.quicksum(Z[i][j] for i in N_PUDO for j in N_PUDO) == 0, name="No transit")
    # mdl.addConstr(gp.quicksum(H[i][j][p] for i in N_PUDO for j in N_td for p in P) == 0, name="No intermodal (DO)")
    # mdl.addConstr(gp.quicksum(H[i][j][p] for i in N_tp for j in N_PUDO for p in P) == 0, name="No intermodal (PU)")
    for j in N_ad:
        mdl.addConstr(gp.quicksum(X[i][j + k * A][v] for i in N_PUDO for k in range(3) for v in V) +
                      gp.quicksum(Y[i][j + k * A] for i in N_PU for k in range(3)) +
                      gp.quicksum(Z[i][j] for i in N_hp + N_ap) == 1, name="Eqn 2")
    for i in N_ap:
        mdl.addConstr(gp.quicksum(X[i + k * A][j][v] for j in N_PUDO for k in range(3) for v in V) +
                      gp.quicksum(Y[i + k * A][j] for j in N_DO for k in range(3)) +
                      gp.quicksum(Z[i][j] for j in N_ad + N_hd) == 1, name="Eqn 3")
    for j in N_ad:
        for k in [1, 2]:
            mdl.addConstr((gp.quicksum(X[i][j + k * A][v] for i in N_PUDO for v in V) +
                           gp.quicksum(Y[i][j + k * A] for i in N_PU)) - Z[j + k * A][j] == 0, name="Eqn 4")
    for i in N_ap:
        for k in [1, 2]:
            mdl.addConstr((gp.quicksum(X[i + k * A][j][v] for j in N_PUDO for v in V) +
                           gp.quicksum(Y[i + k * A][j] for j in N_DO)) - Z[i][i + k * A] == 0, name="Eqn 5")
    for i in N_hat:
        for v in V:
            mdl.addConstr(gp.quicksum(X[i][j][v] for j in N) - gp.quicksum(X[j][i][v] for j in N) == 0, name="Eqn 6")
    for v in V:
        mdl.addConstr(gp.quicksum(X[0][j][v] for j in N_PU) -
                      gp.quicksum(X[i][N_f[0]][v] for i in N_DO) == 0, name="Eqn 7")
    for j in N_hp:
        for v in V:
            mdl.addConstr(gp.quicksum(X[i][j][v] for i in N) -
                          gp.quicksum(X[i][j + k * A][v] for i in N_PUDO for k in [1, 2, 3]) == 0, name="Eqn 8")
    for j in N_hp:
        mdl.addConstr(gp.quicksum(X[i][j + k * A][v] for i in N_PUDO for k in range(1, 4) for v in V) +
                      gp.quicksum(Y[i][j + k * A] for i in N_PU for k in range(1, 4)) +
                      gp.quicksum(Z[i][j + A] for i in N_hp + N_ap) -
                      gp.quicksum(X[j + k * A][i][v] for i in N_PUDO for k in range(11, 14) for v in V) -
                      gp.quicksum(Y[j + k * A][i] for i in N_DO for k in range(11, 14)) -
                      gp.quicksum(Z[j + 11 * A][i] for i in N_ad + N_hd) == 0, name="Eqn 9")
    for v in V:
        for i in N:
            mdl.addConstr(gp.quicksum(X[i][j][v] for j in N_hat) <= 1, name="Eqn 10")
    for v in V:
        mdl.addConstr(gp.quicksum(X[0][j][v] for j in list(set(N).difference(N_PU))) == 0, name="Eqn 11")
    for v in V:
        mdl.addConstr(gp.quicksum(X[i][0][v] for i in N) == 0, name="Eqn 12")
    for v in V:
        mdl.addConstr(gp.quicksum(X[i][N_f[0]][v] for i in list(set(N).difference(N_DO))) == 0, name="Eqn 13")
    for v in V:
        mdl.addConstr(gp.quicksum(X[N_f[0]][j][v] for j in N) == 0, name="Eqn 14")
    for i in N_hp:
        for j in N_hd:
            mdl.addConstr(gp.quicksum(X[i][j][v] for v in V) == 0, name="Eqn 15")
    for i in N:
        mdl.addConstr(gp.quicksum(X[i][j][v] for j in N_hs for v in V) == 0, name="Eqn 16")
    mdl.addConstr(gp.quicksum(Y[i][j] for i in list(set(N).difference(N_PU)) for j in N) == 0, name="Eqn 17")
    mdl.addConstr(gp.quicksum(Y[i][j] for i in N for j in list(set(N).difference(N_DO))) == 0, name="Eqn 18")
    mdl.addConstr(gp.quicksum(Y[i][j] for i in N_hp for j in N_hd) == 0, name="Eqn 19")
    mdl.addConstr(gp.quicksum(Z[i][j] for i in N_ad + N_pk + N_tp + N_hd + N_hs for j in N) == 0, name="Eqn 20")
    mdl.addConstr(gp.quicksum(Z[i][j] for i in N for j in N_hp + N_td + N_pk + N_ap + N_hs) == 0, name="Eqn 21")
    mdl.addConstr(gp.quicksum(Z[i][j] for i in N_hp for j in list(set(N).difference(N_ad))) == 0, name="Eqn 22")

    # Constraint set 2: Temporal constraints on vehicles ============================================================= #
    for i in N_hat:
        for j in N_hat:
            for v in V:
                mdl.addConstr(T[i] + t[i][j] - T[j] <= M * (1 - X[i][j][v]), name="Eqn 23")
    for j in N_PU:
        for v in V:
            mdl.addConstr(T_0[v] - T[j] <= M * (1 - X[0][j][v]), name="Eqn 24")
    for i in N_DO:
        for v in V:
            mdl.addConstr(T[i] - T_f[v] <= M * (1 - X[i][N_f[0]][v]), name="Eqn 25")
    for i in N_PU:
        for j in N_DO:
            mdl.addConstr(T[i] + t[i][j] + w_SAV - T[j] <= M * (1 - Y[i][j]), name="Eqn 26")
    for i in N_hp:
        for k in range(1, 16):
            mdl.addConstr(T[i] + t[i][i + k * A] <= T[i + k * A], name="Eqn 27")
    for i in N_ad:
        mdl.addConstr(T[i] + s[i] <= T[i + 10 * A], name="Eqn 28")
    for i in N_ap:
        mdl.addConstr(T[i] + t[i][i + 3 * A] - T[i + 3 * A] <=
                      M * (1 - gp.quicksum(X[i][i + 3 * A][v] for v in V)), name="Eqn 29")
    for i in N_hp + N_td + N_ap:
        for j in N_ad + N_tp + N_hd:
            mdl.addConstr(T[i] + t_trn[i][j] + t_wlk[i][j] + w_TRN - T[j] <= M * (1 - Z[i][j]), name="Eqn 30")
    for v in V:
        mdl.addConstr(T_0[v] >= a_0, name="Eqn 31")
    for v in V:
        mdl.addConstr(T_f[v] <= b_f, name="Eqn 32")
    for j in N_ad:
        mdl.addConstr(a[j] - T[j] <= M * (1 - gp.quicksum(X[i][j][v] for i in N for v in V) -
                                          gp.quicksum(Y[i][j] for i in N) - gp.quicksum(Z[i][j] for i in N_PUDO)),
                      name="Eqn 33")
    for j in N_ad:
        mdl.addConstr(T[j] - b[j] <= M * (1 - gp.quicksum(X[i][j][v] for i in N for v in V) -
                                          gp.quicksum(Y[i][j] for i in N) - gp.quicksum(Z[i][j] for i in N_PUDO)),
                      name="Eqn 34")
    for i in N_ap:
        mdl.addConstr(a[i] - T[i] <= M * (1 - gp.quicksum(X[i][j][v] for j in N for v in V) -
                                          gp.quicksum(Y[i][j] for j in N) - gp.quicksum(Z[i][j] for j in N_PUDO)),
                      name="Eqn 35")
    for i in N_ap:
        mdl.addConstr(T[i] - b[i] <= M * (1 - gp.quicksum(X[i][j][v] for j in N for v in V) -
                                          gp.quicksum(Y[i][j] for j in N) - gp.quicksum(Z[i][j] for j in N_PUDO)),
                      name="Eqn 36")
    for j in N_ad + N_ap:
        for k in [1, 2]:
            mdl.addConstr((a[j] - t_trn[j + k * A][j] - t_wlk[j + k * A][j] - w_TRN) - T[j + k * A] <=
                          M * (1 - gp.quicksum(X[i][j + k * A][v] for i in N_hat for v in V)), name="Eqn 37")
    for j in N_ad + N_ap:
        for k in [1, 2]:
            mdl.addConstr(T[j + k * A] - (b[j] - t_trn[j + k * A][j] - t_wlk[j + k * A][j] - w_TRN) <=
                          M * (1 - gp.quicksum(X[i][j + k * A][v] for i in N_hat for v in V)), name="Eqn 38")
    for j in N_ad + N_ap:
        for k in [1, 2]:
            mdl.addConstr((a[j] - t_trn[j + k * A][j] - t_wlk[j + k * A][j] - w_TRN) - T[j + k * A] <=
                          M * (1 - gp.quicksum(Y[i][j + k * A] for i in N_hat)), name="Eqn 39")
    for j in N_ad + N_ap:
        for k in [1, 2]:
            mdl.addConstr(T[j + k * A] - (b[j] - t_trn[j + k * A][j] - t_wlk[j + k * A][j] - w_TRN) <=
                          M * (1 - gp.quicksum(Y[i][j + k * A] for i in N_hat)), name="Eqn 40")
    for i in N_hp + N_td + N_ap:
        for j in N_ad:
            mdl.addConstr(a[j] - t_trn[i][j] - t_wlk[i][j] - w_TRN - T[i] <= M * (1 - Z[i][j]), name="Eqn 41")
    for i in N_ap:
        for j in N_ad + N_tp + N_hd:
            mdl.addConstr(T[j] - t_trn[i][j] - t_wlk[i][j] - w_TRN - b[i] <= M * (1 - Z[i][j]), name="Eqn 42")

    # Constraint set 3: Deadheading constraints on vehicles ========================================================== #
    for i in N:
        for j in N_PU:
            for v in V:
                mdl.addConstr(rho[i][v] + 1 - rho[j][v] <= M * (1 - X[i][j][v]), name="Eqn 43")
    for i in N:
        for j in N_PU:
            for v in V:
                mdl.addConstr(rho[j][v] - rho[i][v] - 1 <= M * (1 - X[i][j][v]), name="Eqn 44")
    for i in N:
        for j in N_DO:
            for v in V:
                mdl.addConstr(rho[i][v] - 1 - rho[j][v] <= M * (1 - X[i][j][v]), name="Eqn 45")
    for i in N:
        for j in N_DO:
            for v in V:
                mdl.addConstr(rho[j][v] - rho[i][v] + 1 <= M * (1 - X[i][j][v]), name="Eqn 46")
    for v in V:
        mdl.addConstr(rho[0][v] == 0, name="Eqn 47")
    for i in N_hat:
        for j in N_pk:
            for v in V:
                mdl.addConstr(rho[i][v] + rho[j][v] <= M * (1 - X[i][j][v]), name="Eqn 48")
    for i in N_PUDO:
        for j in N_hat:
            for v in V:
                mdl.addConstr(T[j] - T[i] - t[i][j] - 5 <= M * (1 - X[i][j][v]), name="Eqn 49")
    for v in V:
        mdl.addConstr(gp.quicksum(X[i][j][v] for i in N_hp + N_pk for j in N_pk) == 0, name="Eqn 50")
    for i in N_pk:
        for j in N_hat:
            mdl.addConstr((T[j] - t[i][j] - T[i]) / 60 - K[i] <= M * (1 - gp.quicksum(X[i][j][v] for v in V)),
                          name="Eqn 51")
    for i in N_pk:
        for j in N_hat:
            mdl.addConstr(K[i] - (T[j] - t[i][j] - T[i]) / 60 <= M * (1 - gp.quicksum(X[i][j][v] for v in V)),
                          name="Eqn 52")

    # Constraint set 4: Spatial connectivity constraints on household members ======================================== #
    for p in P:
        for j in theta[p]:
            mdl.addConstr(gp.quicksum(H[i][j + A][p] for i in N_PUDO) == 1, name="Eqn 53")
    for p in P:
        for i in theta[p]:
            mdl.addConstr(gp.quicksum(H[i + 11 * A][j][p] for j in N_PUDO) == 1, name="Eqn 54")
    for p in P:
        for j in theta[p]:
            for k in [2, 3]:
                mdl.addConstr(gp.quicksum(H[i][j + k * A][p] for i in N_PUDO) - H[j + k * A][j + A][p] == 0,
                              name="Eqn 55")
    for p in P:
        for i in theta[p]:
            for k in [12, 13]:
                mdl.addConstr(gp.quicksum(H[i + k * A][j][p] for j in N_PUDO) - H[i + 11 * A][i + k * A][p] == 0,
                              name="Eqn 56")
    for i in N_hat:
        for p in P:
            mdl.addConstr(gp.quicksum(H[i][j][p] for j in N) - gp.quicksum(H[j][i][p] for j in N) == 0, name="Eqn 57")
    for p in P:
        mdl.addConstr(gp.quicksum(H[0][j][p] for j in N_hp) - gp.quicksum(H[i][N_f[0]][p] for i in N_hd) == 0,
                      name="Eqn 58")
    for j in N_hat:
        for p in P:
            mdl.addConstr(gp.quicksum(H[i][j][p] for i in N) <= 1, name="Eqn 59")
    for p in P:
        for j in list(set(N_hat).difference(theta[p])):
            mdl.addConstr(H[0][j][p] == 0, name="Eqn 60")
    for p in P:
        mdl.addConstr(gp.quicksum(H[0][j][p] for j in N) <= 1, name="Eqn 61")
    for p in P:
        mdl.addConstr(gp.quicksum(H[i][0][p] for i in N) == 0, name="Eqn 62")
    for p in P:
        mdl.addConstr(gp.quicksum(H[i][N_f[0]][p] for i in list(set(N).difference(N_hd))) == 0, name="Eqn 63")
    for p in P:
        mdl.addConstr(gp.quicksum(H[N_f[0]][j][p] for j in N) == 0, name="Eqn 64")
    for p in P:
        mdl.addConstr(gp.quicksum(H[i][j][p] for i in N for j in N_pk) == 0, name="Eqn 65")
    for i in N_ad:
        mdl.addConstr(gp.quicksum(H[i][i + 10 * A][p] for p in P) == 1, name="Eqn 66")
    for i in N_ad:
        for j in list(set(N_ap).difference([i + 10 * A])):
            mdl.addConstr(gp.quicksum(H[i][j][p] for p in P) == 0, name="Eqn 67")
    for p in P:
        for j in theta[p]:
            mdl.addConstr(gp.quicksum(H[i][j + 15 * A][p] for i in N_hd) - H[j + 15 * A][j][p] == 0, name="Eqn 68")
    for p in P:
        mdl.addConstr(gp.quicksum(H[i][j][p] for i in N_hp + N_hs for j in N_hs) == 0, name="Eqn 69")
    for j in list(set(N_hat).difference(N_hp)):
        for p in P:
            mdl.addConstr(gp.quicksum(H[i][j][p] for i in N_hs) == 0, name="Eqn 70")
    for p in P:
        for i in list(set(N_hat).difference(N_hs)):
            for j in theta[p]:
                mdl.addConstr(H[i][j][p] == 0, name="Eqn 71")
    for p in P:
        for i in theta[p]:
            for j in list(set(N_hat).difference(N_hs)):
                mdl.addConstr(H[i + 14 * A][j][p] == 0, name="Eqn 72")

    # Constraint set 5: Temporal constraints on household members ==================================================== #
    for p in P:
        for i in N_hat:
            for j in N_hat:
                mdl.addConstr(T[i] + t[i][j] - T[j] <= M * (1 - H[i][j][p]), name="Eqn 73")
    for j in N_hp:
        for p in P:
            mdl.addConstr(T_0[p] - T[j] <= M * (1 - H[0][j][p]), name="Eqn 74")
    for i in N_hd:
        for p in P:
            mdl.addConstr(T[i] - T_f[p] <= M * (1 - H[i][N_f[0]][p]), name="Eqn 75")
    for p in P:
        mdl.addConstr(T_0[p] >= a_0, name="Eqn 76")
    for p in P:
        mdl.addConstr(T_f[p] <= b_f, name="Eqn 77")
    for i in N_hp:
        for k in [1, 2, 3]:
            mdl.addConstr(Q[i] >= T[i + k * A] - t[i][i + k * A] - T[i], name="Eqn 78")
    for i in N_ap:
        mdl.addConstr(Q[i] >= T[i] - (T[i - 10 * A] + s[i - 10 * A]), name="Eqn 79")
    for i in N_ap:
        for k in [1, 2]:
            mdl.addConstr(Q[i + k * A] >=
                          T[i + k * A] - (T[i - 10 * A] + s[i - 10 * A]) -
                          (t_trn[i][i + k * A] + t_wlk[i][i + k * A] + w_TRN), name="Eqn 80")

    # Constraint set 6: Coupling and participation constraints ======================================================= #
    for p in P:
        for i in theta[p]:
            for j in N_hat:
                for k in [0, 11]:
                    mdl.addConstr(H[i + k * A][j][p] - (gp.quicksum(X[i + k * A][j][v] for v in V) +
                                                        Y[i + k * A][j] + Z[i + k * A][j]) == 0, name="Eqn 81")
    for p in P:
        for i in N_hat:
            for j in theta[p]:
                for k in [1, 14]:
                    mdl.addConstr(H[i][j + k * A][p] - (gp.quicksum(X[i][j + k * A][v] for v in V) +
                                                        Y[i][j + k * A] + Z[i][j + k * A]) == 0, name="Eqn 82")
    for p in P:
        for i in theta[p]:
            for j in N_hat:
                for k in [12, 13]:
                    mdl.addConstr(H[i + k * A][j][p] - (gp.quicksum(X[i + k * A][j][v] for v in V) +
                                                        Y[i + k * A][j]) == 0, name="Eqn 83")
    for p in P:
        for i in N_hat:
            for j in theta[p]:
                for k in [2, 3]:
                    mdl.addConstr(H[i][j + k * A][p] - (gp.quicksum(X[i][j + k * A][v] for v in V) +
                                                        Y[i][j + k * A]) == 0, name="Eqn 84")
    for i in N_PU:
        for j in N_hat:
            mdl.addConstr(gp.quicksum(H[i][j][p] for p in P) - 1 <= M * (1 - Y[i][j]), name="Eqn 85")
    for i in N_hat:
        for j in N_DO:
            for p in P:
                mdl.addConstr(1 - (gp.quicksum(X[i][j][v] for v in V) + Y[i][j] + Z[i][j]) <= M * (1 - H[i][j][p]),
                              name="Eqn 86")

    # Model run ====================================================================================================== #
    # Gurobi parameter settings
    if timelimit == "on":
        mdl.Params.TimeLimit = timelimit_v
    if mipgap == "on":
        mdl.Params.MIPGap = mipgap_v
    if incumbent == "on":
        mdl._cur_obj = float('inf')
        mdl._time = time.time()
    if first_solution == "on":
        mdl.Params.SolutionLimit = 1

    print("Now solving...")
    if incumbent == "on":
        mdl.optimize(callback=cb)
    else:
        mdl.optimize()

    if mdl.Status == GRB.OPTIMAL:
        print("Optimal solution found.")
        optimized = str("Yes")
    else:
        print("Optimization terminated without finding an optimal solution.")
        optimized = str("No")

    runtime = round(time.time() - timer[h], 2)
    if incumbent == "on":
        inc_dur = time.time() - mdl._time
    else:
        inc_dur = 0
    print("Runtime =", runtime, "seconds")

    # Model output =================================================================================================== #
    if mdl.objVal != float("inf"):
        print("Exporting solution...")
        print("Total household travel cost =", mdl.objVal)
        sol = mdl.objVal
        gap = mdl.MIPGap
        num_X = 0  # number of PAV trips
        num_Y = 0  # number of SAV trips
        num_Z = 0  # number of transit trips
        num_im = 0  # number of PAV/SAV-transit intermodal trips
        pool_X = 0  # number of PAV-pooling trips
        time_X = 0  # total PAV travel time
        ob_time_X = 0  # total PAV passenger onboard travel time
        time_Y = 0  # total SAV travel time
        time_Z = 0  # total transit travel time
        time_wlk = 0  # total transit access/egress walk time
        time_prk = 0  # total PAV out-of-home parking time
        time_H = 0  # total person travel time except for wait time
        dist_X = 0  # total PAV travel distance
        ob_dist_X = 0  # total PAV passenger onboard travel distance
        pfee_X = 0  # total PAV parking fee
        dist_Y = 0  # total SAV travel distance
        out_prk = 0  # number of out-of-home parks
        fare_Y = 0  # total SAV fare
        fare_Z = 0  # total transit fare
        # X[i][j][v]
        for v in V:
            for i in N:
                for j in N:
                    if X[i][j][v].X > 0.9:
                        res_X = str("X[") + str(v) + str("] = ") + str(X[i][j][v].X)
                        res_T = T[i].X
                        T_hr = res_T // 60
                        T_min = res_T - (T_hr * 60)
                        if i in N_ap + N_tp:
                            res_Q = Q[i].X
                        else:
                            res_Q = str("")
                        res_t = t[i][j]
                        res_t_trn = str("")
                        res_t_wlk = str("")
                        res_rho = rho[i][v].X
                        res_prk = str("")
                        pfee = str("")
                        res_d = d[i][j]
                        oprc = beta_PAVop * res_d
                        fare = str("")
                        if i in N_pk:
                            res_prk = K[i].X
                            pfee = pkf[i] * res_prk
                            if pfee > 0:
                                time_prk += res_prk
                                out_prk += 1
                                pfee_X += pfee
                        record_X = [hhid, res_X, i, j, T_hr, T_min, res_Q, res_t, res_t_trn, res_t_wlk, res_rho,
                                    res_prk, pfee, res_d, oprc, fare]
                        hh_result.append(record_X)
                        num_X += 1
                        dist_X += res_d
                        time_X += res_t
                        if res_rho >= 0.9:
                            ob_dist_X += res_d
                            ob_time_X += res_t
                            if res_rho >= 1.9:
                                pool_X += 1
        # Y[i][j]
        for i in N:
            for j in N:
                if Y[i][j].X > 0.9:
                    res_Y = str("Y =") + str(Y[i][j].X)
                    res_T = T[i].X
                    T_hr = res_T // 60
                    T_min = res_T - (T_hr * 60)
                    res_Q = w_SAV
                    res_t = t[i][j]
                    res_t_trn = str("")
                    res_t_wlk = str("")
                    res_rho = str("")
                    res_prk = str("")
                    pfee = str("")
                    res_d = d[i][j]
                    oprc = str("")
                    fare = beta_SAVfd * res_d + beta_SAVfb
                    record_Y = [hhid, res_Y, i, j, T_hr, T_min, res_Q, res_t, res_t_trn, res_t_wlk, res_rho, res_prk,
                                pfee, res_d, oprc, fare]
                    hh_result.append(record_Y)
                    num_Y += 1
                    dist_Y += res_d
                    time_Y += res_t
                    fare_Y += fare
        # Z[i][j]
        for i in N_PUDO:
            for j in N_PUDO:
                if Z[i][j].X > 0.9:
                    res_Z = str("Z =") + str(Z[i][j].X)
                    res_T = T[i].X
                    T_hr = res_T // 60
                    T_min = res_T - (T_hr * 60)
                    res_Q = w_TRN
                    res_t = str("")
                    res_t_trn = t_trn[i][j]
                    res_t_wlk = t_wlk[i][j]
                    res_rho = str("")
                    res_prk = str("")
                    pfee = str("")
                    res_d = str("")
                    oprc = str("")
                    fare = beta_TRNfr
                    record_Z = [hhid, res_Z, i, j, T_hr, T_min, res_Q, res_t, res_t_trn, res_t_wlk, res_rho, res_prk,
                                pfee, res_d, oprc, fare]
                    hh_result.append(record_Z)
                    num_Z += 1
                    time_Z += res_t_trn
                    fare_Z += fare
                    time_wlk += res_t_wlk
                    if i in N_td or j in N_tp:
                        num_im += 1
        # H[i][j][p]
        for p in P:
            for i in N:
                for j in N:
                    if H[i][j][p].X > 0.9:
                        res_H = str("H[") + str(p) + str("] = ") + str(H[i][j][p].X)
                        res_T = T[i].X
                        T_hr = res_T // 60
                        T_min = res_T - (T_hr * 60)
                        if i in N_ap + N_tp:
                            res_Q = Q[i].X
                        else:
                            res_Q = 0
                        res_t = str("")
                        res_t_trn = str("")
                        res_t_wlk = str("")
                        res_d = str("")
                        oprc = str("")
                        fare = str("")
                        if Z[i][j].X > 0.9:
                            res_Q = w_TRN
                            res_t_trn = t_trn[i][j]
                            res_t_wlk = t_wlk[i][j]
                            fare = beta_TRNfr
                            time_H += res_t_trn + res_t_wlk + res_Q
                        else:
                            res_t = t[i][j]
                            res_d = d[i][j]
                            if Y[i][j].X > 0.9:
                                res_Q = w_SAV
                                fare = beta_SAVfd * res_d + beta_SAVfb
                                time_H += res_t + res_Q
                            for v in V:
                                if X[i][j][v].X > 0.9:
                                    time_H += res_t + res_Q
                        res_rho = str("")
                        res_prk = str("")
                        pfee = str("")
                        record_H = [hhid, res_H, i, j, T_hr, T_min, res_Q, res_t, res_t_trn, res_t_wlk, res_rho,
                                    res_prk, pfee, res_d, oprc, fare]
                        hh_result.append(record_H)
        print("")
        print("Household", hhid, "travel planning completed.")

        term1 = beta_PAVwt * gp.quicksum(Q[i] for i in N_ap + N_tp).getValue()
        term2 = beta_PAViv * gp.quicksum(t[i][j] * H[i][j][p] for i in N for j in N for p in P).getValue()
        term3 = beta_PAVop * gp.quicksum(d[i][j] * X[i][j][v] for i in N for j in N for v in V).getValue()
        term4 = gp.quicksum(pkf[i] * K[i] for i in N_pk).getValue()
        term5 = beta_SAVwt * gp.quicksum(w_SAV * Y[i][j] for i in N for j in N).getValue()
        term6 = (beta_SAViv - beta_PAViv) * gp.quicksum(t[i][j] * Y[i][j] for i in N for j in N).getValue()
        term7 = beta_SAVfb * gp.quicksum(Y[i][j] for i in N for j in N).getValue()
        term8 = beta_SAVfd * gp.quicksum(d[i][j] * Y[i][j] for i in N for j in N).getValue()
        term9 = beta_TRNwk * gp.quicksum(t_wlk[i][j] * Z[i][j] for i in N_PUDO for j in N_PUDO).getValue()
        term10 = beta_TRNwt * gp.quicksum(w_TRN * Z[i][j] for i in N_PUDO for j in N_PUDO).getValue()
        term11 = beta_TRNiv * gp.quicksum((t_trn[i][j] - t[i][j]) * Z[i][j] for i in N_PUDO for j in N_PUDO).getValue()
        term12 = beta_TRNfr * gp.quicksum(Z[i][j] for i in N_PUDO for j in N_PUDO).getValue()

        # Household results
        hh_obj = [str("HH cost"), str(sol), "", "", "", "", "", "", "", "", "", "", "", "", "", ""]
        hh_run = [str("Runtime"), str(runtime), "", "", "", "", "", "", "", "", "", "", "", "", "", ""]
        hh_nac = [str("Activities"), str(A), "", "", "", "", "", "", "", "", "", "", "", "", "", ""]
        hh_spc = ["", "", "", "", "", "", "", "", "", "", "", "", "", "", "", ""]
        hh_t_1 = [str("PAVwt"), str(term1), "", "", "", "", "", "", "", "", "", "", "", "", "", ""]
        hh_t_2 = [str("PAViv"), str(term2), "", "", "", "", "", "", "", "", "", "", "", "", "", ""]
        hh_t_3 = [str("PAVop"), str(term3), "", "", "", "", "", "", "", "", "", "", "", "", "", ""]
        hh_t_4 = [str("PAVpk"), str(term4), "", "", "", "", "", "", "", "", "", "", "", "", "", ""]
        hh_t_5 = [str("SAVwt"), str(term5), "", "", "", "", "", "", "", "", "", "", "", "", "", ""]
        hh_t_6 = [str("SAViv"), str(term6), "", "", "", "", "", "", "", "", "", "", "", "", "", ""]
        hh_t_7 = [str("SAVfb"), str(term7), "", "", "", "", "", "", "", "", "", "", "", "", "", ""]
        hh_t_8 = [str("SAVfd"), str(term8), "", "", "", "", "", "", "", "", "", "", "", "", "", ""]
        hh_t_9 = [str("TRNwk"), str(term9), "", "", "", "", "", "", "", "", "", "", "", "", "", ""]
        hh_t_10 = [str("TRNwt"), str(term10), "", "", "", "", "", "", "", "", "", "", "", "", "", ""]
        hh_t_11 = [str("TRNiv"), str(term11), "", "", "", "", "", "", "", "", "", "", "", "", "", ""]
        hh_t_12 = [str("TRNfr"), str(term12), "", "", "", "", "", "", "", "", "", "", "", "", "", ""]
        hh_result.append(hh_obj)
        hh_result.append(hh_run)
        hh_result.append(hh_nac)
        hh_result.append(hh_spc)
        hh_result.append(hh_t_1)
        hh_result.append(hh_t_2)
        hh_result.append(hh_t_3)
        hh_result.append(hh_t_4)
        hh_result.append(hh_t_5)
        hh_result.append(hh_t_6)
        hh_result.append(hh_t_7)
        hh_result.append(hh_t_8)
        hh_result.append(hh_t_9)
        hh_result.append(hh_t_10)
        hh_result.append(hh_t_11)
        hh_result.append(hh_t_12)
        print("")
        print("Exporting household results...")
        hh_column = ['HHID', 'DV', "From", "To", 'T_hr', 'T_min', 'Wt_time', 'AV_time', 'Trn_time', 'Wlk_time',
                     'Onboard', 'Prk_time', 'Prk_fee', 'AV_dist', 'PAV_opr', 'Fare']
        df_hh_result = pd.DataFrame(hh_result, columns=hh_column)
        df_hh_result.to_csv(r'results\sce' + str(hh_sce) + str(nw_sce) + '_hh' + str(hhid) + '_y' + str(num_Y) + '_z'
                            + str(num_Z) + '_im' + str(num_im) + '.csv', index=False)

        # Total results
        record_HH = [{'hhid': hhid, 'hh_size': len(P), 'activities': A, 'PAVs': len(V), 'PAV_trips': num_X,
                      'SAV_trips': num_Y, 'trn_trips': num_Z, 'im_trips:': num_im, 'PAV_pooling': pool_X,
                      'PAV_time': time_X, 'ob_PAV_time': ob_time_X, 'dh_PAV_time': time_X - ob_time_X,
                      'SAV_time': time_Y, 'trn_time': time_Z, 'prk_time': time_prk * 60, 'wlk_time': time_wlk,
                      'prs_time': round(time_H, 0), 'PAV_dist': dist_X, 'ob_PAV_dist': ob_dist_X,
                      'dh_PAV_dist': dist_X - ob_dist_X, 'SAV_dist': dist_Y, 'VKT': round((dist_X + dist_Y) / 1000, 1),
                      'out_HL_prk': out_prk, 'prk_fee': pfee_X, 'SAV_fare': fare_Y, 'trn_fare': fare_Z,
                      'runtime': round(time.time() - timer[h], 2), 'obj_value': sol, 'gap': gap, 'sol_dur': inc_dur,
                      'optimized?': optimized}]
        df_sce_result = pd.DataFrame(record_HH)
        df_sce_result.to_csv('result_sce' + str(hh_sce) + str(nw_sce) + '.csv', mode='a', index=False, header=False)
        print("")
    else:
        print("No feasible solution found.")
        print("Moving to the next household...")
        print("")

# ==================================================================================================================== #
# TERMINATION
# ==================================================================================================================== #
print("")
print("HOUSEHOLD AV TRAVEL PLANNING IS COMPLETED!")

print("Total runtime =", round(time.time() - start_time, 2), "seconds")
