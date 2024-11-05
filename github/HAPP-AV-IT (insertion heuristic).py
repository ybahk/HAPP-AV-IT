# ==================================================================================================================== #
# |||                                      * HAPP-AV-IT Insertion Heuristic *                                      ||| #
# |||                                                                                                              ||| #
# |||                                          Developed by Younghun Bahk                                          ||| #
# |||                                    Version 1.01 / Last Update: 2024.10.03.                                   ||| #
# ==================================================================================================================== #

import time
import pandas as pd
import networkx as nx
import pickle
from collections import OrderedDict
from itertools import permutations
from itertools import product
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
timelimit_v = 172800
mipgap = "off"  # stop when gap tolerance reached
mipgap_v = 0.1
incumbent = "off"  # stop when maximum incumbent time reached
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
ih_column = ['hhid', 'hh_size', 'activities', 'PAVs', 'runtime', 'obj_value']
df_ih_result = pd.DataFrame(columns=ih_column)
df_ih_result.to_csv(r'result_ih.csv', index=False)

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
    print("* HOUSEHOLD AV TRAVEL PLANNING MODEL: INSERTION HEURISTIC VERSION")

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

    # Step 1: Feasibility check ====================================================================================== #
    feasible_schedules = OrderedDict()
    extended_schedules = OrderedDict()
    F = OrderedDict()
    time_pav = OrderedDict()
    time_trn = OrderedDict()
    time_TRV = OrderedDict()
    act_TRV = OrderedDict()
    T_p = OrderedDict()
    flexible = OrderedDict()
    for p in reversed(P):
        if len(theta[p]) == 0:
            P.remove(P[-1])
    for p in P:
        # Feasibility check 1
        feasible_schedules[p] = []
        extended_schedules[p] = []
        F[p] = []
        time_pav[p] = []
        time_trn[p] = []
        time_TRV[p] = []
        act_TRV[p] = []
        T_p[p] = []
        if len(theta[p]) > 1:
            for sequence in list(permutations(theta[p])):
                current_time = 0
                is_feasible = True
                for act_p in list(sequence):
                    start = max(a[act_p + A], current_time)
                    end_time = max(start + s[act_p + A], a[act_p + 11 * A])
                    if start > b[act_p + A] or end_time > b[act_p + 11 * A]:
                        is_feasible = False
                        break
                    current_time = end_time
                if is_feasible:
                    feasible_schedules[p].append(act_p for act_p in list(sequence))
            for schedule in feasible_schedules[p]:
                schedule = list(schedule)
                n = len(schedule)
                for i in range(2 ** (n - 1)):
                    new_sequence = [schedule[0]]
                    for j in range(n - 1):
                        if (i >> j) & 1:
                            new_sequence.append("h")
                        new_sequence.append(schedule[j + 1])
                    extended_schedules[p].append(new_sequence)
            for schedule in extended_schedules[p]:
                F[p].append(schedule)
        elif len(theta[p]) == 1:
            F[p].append(theta[p])
        # Travel time check
        for k in range(len(F[p])):
            time_pav[p].append([])
            time_trn[p].append([])
            time_TRV[p].append([])
            act_TRV[p].append([])
            T_p[p].append([])
            for l in range(len(F[p][k]) + 1):
                if l == 0 or F[p][k][l - 1] == "h":
                    time_pav[p][k].append(t[F[p][k][l]][F[p][k][l] + A])
                    time_trn[p][k].append(t_wlk[F[p][k][l]][F[p][k][l] + A] + t_trn[F[p][k][l]][F[p][k][l] + A] + w_TRN)
                    time_TRV[p][k].append(min(time_pav[p][k][l], time_trn[p][k][l]))
                    act_TRV[p][k].append([F[p][k][l], F[p][k][l] + A])
                elif l == len(F[p][k]) or F[p][k][l] == "h":
                    time_pav[p][k].append(t[F[p][k][l - 1] + 11 * A][F[p][k][l - 1] + 14 * A])
                    time_trn[p][k].append(t_wlk[F[p][k][l - 1] + 11 * A][F[p][k][l - 1] + 14 * A] +
                                          t_trn[F[p][k][l - 1] + 11 * A][F[p][k][l - 1] + 14 * A] + w_TRN)
                    time_TRV[p][k].append(min(time_pav[p][k][l], time_trn[p][k][l]))
                    act_TRV[p][k].append([F[p][k][l - 1] + 11 * A, F[p][k][l - 1] + 14 * A])
                else:
                    time_pav[p][k].append(t[F[p][k][l - 1] + 11 * A][F[p][k][l] + A])
                    time_trn[p][k].append(t_wlk[F[p][k][l - 1] + 11 * A][F[p][k][l] + A] +
                                          t_trn[F[p][k][l - 1] + 11 * A][F[p][k][l] + A] + w_TRN)
                    time_TRV[p][k].append(min(time_pav[p][k][l], time_trn[p][k][l]))
                    act_TRV[p][k].append([F[p][k][l - 1] + 11 * A, F[p][k][l] + A])
        # Feasibility check 2
        if len(theta[p]) > 1:
            infeasible = []
            for k in reversed(range(len(F[p]))):
                T_p[p][k] = OrderedDict()
                current_time = 0
                is_feasible = True
                for l in range(len(F[p][k])):
                    if F[p][k][l] != "h":
                        act_p = F[p][k][l]
                        prev_act_p = F[p][k][l - 1]
                        # if prev_act_p != "h":
                        travel_time = time_TRV[p][k][l]
                        current_time += travel_time
                        start = max(a[act_p + A], current_time)
                        end_time = max(start + s[act_p + A], a[act_p + 11 * A])
                        if start > b[act_p + A] or end_time > b[act_p + 11 * A]:
                            is_feasible = False
                            break
                        T_p[p][k][act_p] = start
                        current_time = end_time
                if not is_feasible:
                    F[p].remove(F[p][k])
                    time_TRV[p].remove(time_TRV[p][k])
                    time_pav[p].remove(time_pav[p][k])
                    time_trn[p].remove(time_trn[p][k])
                    act_TRV[p].remove(act_TRV[p][k])
                    T_p[p].remove(T_p[p][k])
        else:
            for k in reversed(range(len(F[p]))):
                T_p[p][k] = OrderedDict()
                T_p[p][k][theta[p][0]] = a[theta[p][0] + A]
        # Flexibility check
        flexible[p] = OrderedDict()
        for k in range(len(F[p])):
            flexible[p][k] = OrderedDict()
            for l in range(len(F[p][k])):
                if F[p][k][l] != "h":
                    temp_flex = []
                    for n in range(len(F[p][k])):
                        if F[p][k][n] != "h":
                            act_p = F[p][k][n]
                            if l <= n:
                                temp_flex.append(b[act_p + A] - T_p[p][k][act_p])
                    flexible[p][k][F[p][k][l]] = min(temp_flex)
    act_TRV_com = []
    all_com = product(*([act_TRV[p] for p in P]))
    for combination in all_com:
        merged_sequence = []
        for sequence in combination:
            merged_sequence += sequence
        act_TRV_com.append(merged_sequence)
    unit = OrderedDict()
    for p in reversed(P):
        if p == len(P):
            unit[p] = 1
        else:
            unit[p] = unit[p + 1] * len(act_TRV[p + 1])
    unit[0] = unit[1] * len(act_TRV[1])

    # Step 2: Vehicle insertions ===================================================================================== #
    # Travel cost calculation
    M = ["PAV", "SAV", "Trn", "P_T", "S_T"]
    cost_TRV = OrderedDict()
    save_by_pav = []
    sorted_modes = []
    sorted_l_cost = []
    better_P_T = OrderedDict()
    better_S_T = OrderedDict()
    for k in range(len(act_TRV_com)):
        save_by_pav.append([])
        sorted_modes.append([])
        sorted_l_cost.append([])
        cost_TRV[k] = OrderedDict()
        better_P_T[k] = OrderedDict()
        better_S_T[k] = OrderedDict()
        for l in range(len(act_TRV_com[k])):
            cost_TRV[k][l] = OrderedDict()
            i = act_TRV_com[k][l][0]
            j = act_TRV_com[k][l][1]
            save_by_pav[k].append([])
            sorted_modes[k].append([])
            for m in M:
                if m == "PAV":
                    cost_TRV[k][l][m] = beta_PAViv * t[i][j] + beta_PAVop * d[i][j]
                elif m == "SAV":
                    cost_TRV[k][l][m] = beta_SAVwt * w_SAV + beta_SAViv * t[i][j] + beta_SAVfb + beta_SAVfd * d[i][j]
                elif m == "Trn":
                    cost_TRV[k][l][m] = beta_TRNwk * t_wlk[i][j] + beta_TRNwt * w_TRN + \
                                        beta_TRNiv * t_trn[i][j] + beta_TRNfr
                elif m == "P_T":
                    if 0 < i <= A < j <= 2 * A:
                        P_T_option_1 = beta_PAViv * t[i][i + 2 * A] + beta_PAVop * d[i][i + 2 * A] + \
                                       beta_TRNwk * t_wlk[i + 2 * A][i + A] + beta_TRNwt * w_TRN + \
                                       beta_TRNiv * t_trn[i + 2 * A][i + A] + beta_TRNfr
                        P_T_option_2 = beta_PAViv * t[i][i + 3 * A] + beta_PAVop * d[i][i + 3 * A] + \
                                       beta_TRNwk * t_wlk[i + 3 * A][i + A] + beta_TRNwt * w_TRN + \
                                       beta_TRNiv * t_trn[i + 3 * A][i + A] + beta_TRNfr
                        if P_T_option_1 <= P_T_option_2:
                            cost_TRV[k][l][m] = P_T_option_1
                            better_P_T[k][l] = "option 1"
                        else:
                            cost_TRV[k][l][m] = P_T_option_2
                            better_P_T[k][l] = "option 2"
                    elif 11 * A < i <= 12 * A and 14 * A < j <= 15 * A:
                        P_T_option_1 = beta_PAViv * t[i + A][i + 3 * A] + beta_PAVop * d[i + A][i + 3 * A] + \
                                       beta_TRNwk * t_wlk[i][i + A] + beta_TRNwt * w_TRN + \
                                       beta_TRNiv * t_trn[i][i + A] + beta_TRNfr
                        P_T_option_2 = beta_PAViv * t[i + 2 * A][i + 3 * A] + beta_PAVop * d[i + 2 * A][i + 3 * A] + \
                                       beta_TRNwk * t_wlk[i][i + 2 * A] + beta_TRNwt * w_TRN + \
                                       beta_TRNiv * t_trn[i][i + 2 * A] + beta_TRNfr
                        if P_T_option_1 <= P_T_option_2:
                            cost_TRV[k][l][m] = P_T_option_1
                            better_P_T[k][l] = "option 1"
                        else:
                            cost_TRV[k][l][m] = P_T_option_2
                            better_P_T[k][l] = "option 2"
                    else:
                        cost_TRV[k][l][m] = 9999
                elif m == "S_T":
                    if 0 < i <= A < j <= 2 * A:
                        S_T_option_1 = beta_SAVwt * w_SAV + beta_SAViv * t[i][i + 2 * A] + \
                                       beta_SAVfb + beta_SAVfd * d[i][i + 2 * A] + \
                                       beta_TRNwk * t_wlk[i + 2 * A][i + A] + beta_TRNwt * w_TRN + \
                                       beta_TRNiv * t_trn[i + 2 * A][i + A] + beta_TRNfr
                        S_T_option_2 = beta_SAVwt * w_SAV + beta_SAViv * t[i][i + 3 * A] + \
                                       beta_SAVfb + beta_SAVfd * d[i][i + 3 * A] + \
                                       beta_TRNwk * t_wlk[i + 3 * A][i + A] + beta_TRNwt * w_TRN + \
                                       beta_TRNiv * t_trn[i + 3 * A][i + A] + beta_TRNfr
                        cost_TRV[k][l][m] = min(S_T_option_1, S_T_option_2)
                        if S_T_option_1 <= S_T_option_2:
                            cost_TRV[k][l][m] = S_T_option_1
                            better_S_T[k][l] = "option 1"
                        else:
                            cost_TRV[k][l][m] = S_T_option_2
                            better_S_T[k][l] = "option 2"
                    elif 11 * A < i <= 12 * A and 14 * A < j <= 15 * A:
                        S_T_option_1 = beta_SAVwt * w_SAV + beta_SAViv * t[i + A][i + 3 * A] + \
                                       beta_SAVfb + beta_SAVfd * d[i + A][i + 3 * A] + \
                                       beta_TRNwk * t_wlk[i][i + A] + beta_TRNwt * w_TRN + \
                                       beta_TRNiv * t_trn[i][i + A] + beta_TRNfr
                        S_T_option_2 = beta_SAVwt * w_SAV + beta_SAViv * t[i + 2 * A][i + 3 * A] + \
                                       beta_SAVfb + beta_SAVfd * d[i + 2 * A][i + 3 * A] + \
                                       beta_TRNwk * t_wlk[i][i + 2 * A] + beta_TRNwt * w_TRN + \
                                       beta_TRNiv * t_trn[i][i + 2 * A] + beta_TRNfr
                        cost_TRV[k][l][m] = min(S_T_option_1, S_T_option_2)
                        if S_T_option_1 <= S_T_option_2:
                            cost_TRV[k][l][m] = S_T_option_1
                            better_S_T[k][l] = "option 1"
                        else:
                            cost_TRV[k][l][m] = S_T_option_2
                            better_S_T[k][l] = "option 2"
                    else:
                        cost_TRV[k][l][m] = 9999
            save_by_pav[k][l] = min(cost_TRV[k][l]["SAV"], cost_TRV[k][l]["Trn"], cost_TRV[k][l]["P_T"],
                                    cost_TRV[k][l]["S_T"]) - cost_TRV[k][l]["PAV"]
            sorted_modes[k][l] = sorted(cost_TRV[k][l].keys(), key=lambda mode: cost_TRV[k][l][mode])
        sorted_l_cost[k] = sorted(range(len(save_by_pav[k])), key=lambda l: save_by_pav[k][l])
    # PAV insertion
    total_cost = []
    mode_TRV = []
    T_v = OrderedDict()
    L_v = OrderedDict()
    lth = OrderedDict()
    ocpy = OrderedDict()
    displaced = OrderedDict()
    L_pk = OrderedDict()
    k_adj = OrderedDict()
    sorted_indices = OrderedDict()
    who = OrderedDict()
    pooled_TRV = OrderedDict()
    is_pooled = OrderedDict()
    dh_cost = OrderedDict()
    route_PAV = OrderedDict()
    wait_pav = OrderedDict()
    PAV_dist = 0
    PAV_time = 0
    for k in range(len(act_TRV_com)):
        mode_TRV.append([])
        total_cost.append([])
        total_cost[k] = 0
        temp_cost = 0
        dh_cost_before = 0
        dh_cost_after = 0
        T_v[k] = OrderedDict()
        L_v[k] = OrderedDict()
        lth[k] = OrderedDict()
        ocpy[k] = OrderedDict()
        displaced[k] = OrderedDict()
        L_pk[k] = []
        who[k] = OrderedDict()
        route_PAV[k] = OrderedDict()
        if len(V) != 0:
            # Initial PAV travel insertion
            init_l = sorted_l_cost[k][0]
            for l in range(len(act_TRV_com[k])):
                mode_TRV[k].append([])
                i = act_TRV_com[k][l][0]
                j = act_TRV_com[k][l][1]
                if l == init_l:
                    mode_TRV[k][l] = "PAV"
                    new_pav_TRV = 1
                    for v in V:
                        T_v[k][v] = []
                        L_v[k][v] = []
                        lth[k][v] = []
                        if new_pav_TRV == 1:
                            if (j - 1) // A == 1:  # activity drop-off
                                for p in P:
                                    k_adj[p] = k % unit[p - 1] // unit[p]
                                    if j % A == 0:
                                        act_p = A
                                    else:
                                        act_p = j % A
                                    if act_p in T_p[p][k_adj[p]]:
                                        T_v[k][v].append([T_p[p][k_adj[p]][act_p] - t[i][j],
                                                          T_p[p][k_adj[p]][act_p]])
                                        L_v[k][v].append([i, j])
                                        lth[k][v].append(l)
                            elif (i - 1) // A == 11:  # activity pickup
                                for p in P:
                                    k_adj[p] = k % unit[p - 1] // unit[p]
                                    if i % A == 0:
                                        act_p = A
                                    else:
                                        act_p = i % A
                                    if act_p in T_p[p][k_adj[p]]:
                                        pickup_time = max(T_p[p][k_adj[p]][act_p] + s[act_p + A], a[act_p + 11 * A])
                                        T_v[k][v].append([pickup_time, pickup_time + t[i][j]])
                                        L_v[k][v].append([i, j])
                                        lth[k][v].append(l)
                            new_pav_TRV = 0
                    total_cost[k] += cost_TRV[k][l]["PAV"]
                else:
                    if sorted_modes[k][l][0] == "PAV" or sorted_modes[k][l][0] == "P_T":
                        if sorted_modes[k][l][1] == "P_T" or sorted_modes[k][l][1] == "PAV":
                            mode_TRV[k][l] = sorted_modes[k][l][2]
                        else:
                            mode_TRV[k][l] = sorted_modes[k][l][1]
                    else:
                        mode_TRV[k][l] = sorted_modes[k][l][0]
                    total_cost[k] += cost_TRV[k][l][mode_TRV[k][l]]
            for l in sorted_l_cost[k]:
                if l == init_l:
                    continue
                elif save_by_pav[k][l] > 0:
                    i = act_TRV_com[k][l][0]
                    j = act_TRV_com[k][l][1]
                    if cost_TRV[k][l]["PAV"] < cost_TRV[k][l]["P_T"]:  # inserting PAV
                        new_pav_TRV = 1
                        for v in V:
                            if new_pav_TRV == 1:
                                if (j - 1) // A == 1 and (i - 1) // A == 11:
                                    for p in P:
                                        k_adj[p] = k % unit[p - 1] // unit[p]
                                        if j % A == 0:
                                            act_p = A
                                        else:
                                            act_p = j % A
                                        if act_p in T_p[p][k_adj[p]]:
                                            dropoff_time = T_p[p][k_adj[p]][act_p]
                                    for p in P:
                                        k_adj[p] = k % unit[p - 1] // unit[p]
                                        if i % A == 0:
                                            act_p = A
                                        else:
                                            act_p = i % A
                                        if act_p in T_p[p][k_adj[p]]:
                                            pickup_time = max(T_p[p][k_adj[p]][act_p] + s[act_p + A], a[act_p + 11 * A])
                                    if pickup_time + t[i][j] < dropoff_time:
                                        if not any(T_v[k][v][n][0] < dropoff_time and pickup_time < T_v[k][v][n][1]
                                                   for n in range(len(T_v[k][v]))):
                                            T_v[k][v].append([pickup_time, dropoff_time])
                                            L_v[k][v].append([i, j])
                                            lth[k][v].append(l)
                                            if j % A == 0:
                                                act_p = A
                                            else:
                                                act_p = j % A
                                            total_cost[k] += (beta_PAViv * (t[i][act_p + 10 * A] +
                                                                            t[act_p + 10 * A][j]) +
                                                              beta_PAVop * (d[i][act_p + 10 * A] +
                                                                            d[act_p + 10 * A][j]))
                                            total_cost[k] -= cost_TRV[k][l][mode_TRV[k][l]]
                                            L_pk[k].append(act_p + 10 * A)
                                            displaced[k][l] = mode_TRV[k][l]
                                            mode_TRV[k][l] = "PAV"
                                            new_pav_TRV = 0
                                elif (j - 1) // A == 1:  # activity drop-off
                                    for p in P:
                                        k_adj[p] = k % unit[p - 1] // unit[p]
                                        if j % A == 0:
                                            act_p = A
                                        else:
                                            act_p = j % A
                                        if act_p in T_p[p][k_adj[p]]:
                                            ocpy[k][l] = [T_p[p][k_adj[p]][act_p] - t[i][j],
                                                          T_p[p][k_adj[p]][act_p]]
                                            if not any(T_v[k][v][n][0] < ocpy[k][l][1] and
                                                       ocpy[k][l][0] < T_v[k][v][n][1] for n in range(len(T_v[k][v]))):
                                                T_v[k][v].append(ocpy[k][l])
                                                L_v[k][v].append([i, j])
                                                lth[k][v].append(l)
                                                total_cost[k] += cost_TRV[k][l]["PAV"]
                                                total_cost[k] -= cost_TRV[k][l][mode_TRV[k][l]]
                                                displaced[k][l] = mode_TRV[k][l]
                                                mode_TRV[k][l] = "PAV"
                                                new_pav_TRV = 0
                                elif (i - 1) // A == 11:  # activity pickup
                                    for p in P:
                                        k_adj[p] = k % unit[p - 1] // unit[p]
                                        if i % A == 0:
                                            act_p = A
                                        else:
                                            act_p = i % A
                                        if act_p in T_p[p][k_adj[p]]:
                                            pickup_time = max(T_p[p][k_adj[p]][act_p] + s[act_p + A], a[act_p + 11 * A])
                                            ocpy[k][l] = [pickup_time, pickup_time + t[i][j]]
                                            if not any(T_v[k][v][n][0] < ocpy[k][l][1] and
                                                       ocpy[k][l][0] < T_v[k][v][n][1] for n in range(len(T_v[k][v]))):
                                                T_v[k][v].append(ocpy[k][l])
                                                L_v[k][v].append([i, j])
                                                lth[k][v].append(l)
                                                total_cost[k] += cost_TRV[k][l]["PAV"]
                                                total_cost[k] -= cost_TRV[k][l][mode_TRV[k][l]]
                                                displaced[k][l] = mode_TRV[k][l]
                                                mode_TRV[k][l] = "PAV"
                                                new_pav_TRV = 0
                    else:  # inserting P_T
                        new_pav_TRV = 1
                        for v in V:
                            if new_pav_TRV == 1:
                                if (j - 1) // A == 1:  # activity drop-off
                                    for p in P:
                                        k_adj[p] = k % unit[p - 1] // unit[p]
                                        if j % A == 0:
                                            act_p = A
                                        else:
                                            act_p = j % A
                                        if act_p in T_p[p][k_adj[p]]:
                                            if better_P_T[k][l] == "option 1":
                                                ocpy[k][l] = [T_p[p][k_adj[p]][act_p] -
                                                              (t[i][j + A] + t_trn[j + A][j]),
                                                              T_p[p][k_adj[p]][act_p]]
                                            elif better_P_T[k][l] == "option 2":
                                                ocpy[k][l] = [T_p[p][k_adj[p]][act_p] -
                                                              (t[i][j + 2 * A] + t_trn[j + 2 * A][j]),
                                                              T_p[p][k_adj[p]][act_p]]
                                            else:
                                                continue
                                            if not any(T_v[k][v][n][0] < ocpy[k][l][1] and ocpy[k][l][0] <
                                                       T_v[k][v][n][1] for n in range(len(T_v[k][v]))):
                                                T_v[k][v].append(ocpy[k][l])
                                                L_v[k][v].append([i, j])
                                                lth[k][v].append(l)
                                                total_cost[k] += cost_TRV[k][l]["P_T"]
                                                total_cost[k] -= cost_TRV[k][l][mode_TRV[k][l]]
                                                displaced[k][l] = mode_TRV[k][l]
                                                mode_TRV[k][l] = "P_T"
                                                new_pav_TRV = 0
                                if (i - 1) // A == 11:  # activity pickup
                                    for p in P:
                                        k_adj[p] = k % unit[p - 1] // unit[p]
                                        if i % A == 0:
                                            act_p = A
                                        else:
                                            act_p = i % A
                                        if act_p in T_p[p][k_adj[p]]:
                                            if better_P_T[k][l] == "option 1":
                                                pickup_time = max(T_p[p][k_adj[p]][act_p] + s[act_p + A],
                                                                  a[act_p + 11 * A])
                                                ocpy[k][l] = [pickup_time + t_trn[i][i + A],
                                                              pickup_time + t_trn[i][i + A] + t[i + A][j]]
                                            elif better_P_T[k][l] == "option 2":
                                                pickup_time = max(T_p[p][k_adj[p]][act_p] + s[act_p + A],
                                                                  a[act_p + 11 * A])
                                                ocpy[k][l] = [pickup_time + t_trn[i][i + 2 * A],
                                                              pickup_time + t_trn[i][i + 2 * A] + t[i + A][j]]
                                            else:
                                                continue
                                            if not any(T_v[k][v][n][0] < ocpy[k][l][1] and ocpy[k][l][0] <
                                                       T_v[k][v][n][1] for n in range(len(T_v[k][v]))):
                                                T_v[k][v].append(ocpy[k][l])
                                                L_v[k][v].append([i, j])
                                                lth[k][v].append(l)
                                                total_cost[k] += cost_TRV[k][l]["P_T"]
                                                total_cost[k] -= cost_TRV[k][l][mode_TRV[k][l]]
                                                displaced[k][l] = mode_TRV[k][l]
                                                mode_TRV[k][l] = "P_T"
                                                new_pav_TRV = 0
                else:  # if neither PAV nor P_T are beneficial
                    continue
            # Pooling availability check
            sorted_indices[k] = OrderedDict()
            sorted_l_time = OrderedDict()
            pooled_TRV[k] = []
            is_pooled[k] = []
            wait_pav[k] = 0
            for v in V:
                if len(T_v[k][v]) >= 1:
                    sorted_indices[k][v] = sorted(range(len(T_v[k][v])), key=lambda i: T_v[k][v][i][0])
                    sorted_l_time[v] = [lth[k][v][i] for i in sorted_indices[k][v]]
            for l in range(len(act_TRV_com[k])):
                i = act_TRV_com[k][l][0]
                j = act_TRV_com[k][l][1]
                if i % A == 0:
                    act_p = A
                else:
                    act_p = i % A
                for p in P:
                    k_adj[p] = k % unit[p - 1] // unit[p]
                    if act_p in theta[p]:
                        who[k][l] = p
            for l in range(len(act_TRV_com[k])):
                l_veh = 0
                i = act_TRV_com[k][l][0]
                j = act_TRV_com[k][l][1]
                if i % A == 0:
                    act_i = A
                else:
                    act_i = i % A
                if j % A == 0:
                    act_j = A
                else:
                    act_j = j % A
                for v in V:
                    if [i, j] in L_v[k][v]:
                        l_veh = v
                if mode_TRV[k][l] == "PAV" and [i, j] not in is_pooled[k]:
                    for n in range(len(act_TRV_com[k])):
                        n_veh = 0
                        n_i = act_TRV_com[k][n][0]
                        n_j = act_TRV_com[k][n][1]
                        pl_code = "none"
                        time_diff = 0
                        if n_i % A == 0:
                            act_n_i = A
                        else:
                            act_n_i = n_i % A
                        if n_j % A == 0:
                            act_n_j = A
                        else:
                            act_n_j = n_j % A
                        for v in V:
                            if [n_i, n_j] in L_v[k][v]:
                                n_veh = v
                        if who[k][n] != who[k][l] and ([n_i, n_j] not in is_pooled[k]) and ([i, j] not in is_pooled[k]):
                            prev_cost_ob = cost_TRV[k][l][mode_TRV[k][l]] + cost_TRV[k][n][mode_TRV[k][n]]
                            if mode_TRV[k][n] == "PAV" or mode_TRV[k][n] == "P_T":
                                prev_cost = prev_cost_ob + beta_PAVop * d[j][n_i]
                            else:
                                prev_cost = prev_cost_ob
                            pl_cost = OrderedDict()
                            pl_type = OrderedDict()
                            pl_type_selected = "none"
                            new_mode = OrderedDict()
                            new_mode_selected = "none"
                            pl_cost[0] = pl_cost[1] = pl_cost[2] = pl_cost[3] = pl_cost[4] = pl_cost[5] = pl_cost[6] = \
                                pl_cost[7] = pl_cost[8] = pl_cost[9] = pl_cost[10] = pl_cost[11] = pl_cost[12] = \
                                pl_cost[13] = pl_cost[14] = pl_cost[15] = pl_cost[16] = pl_cost[17] = pl_cost[18] = \
                                pl_cost[19] = pl_cost[20] = pl_cost[21] = pl_cost[22] = pl_cost[23] = pl_cost[24] = \
                                pl_cost[25] = pl_cost[26] = pl_cost[27] = pl_cost[28] = pl_cost[29] = 9999
                            a1_flex = flexible[who[k][l]][k_adj[who[k][l]]][act_i]
                            a1_early = min(T_p[who[k][l]][k_adj[who[k][l]]][act_i] + s[act_i + A],
                                           a[act_i + 11 * A])
                            a1_late = max(T_p[who[k][l]][k_adj[who[k][l]]][act_i] + s[act_i + A],
                                           a[act_i + 11 * A])
                            a2_flex = flexible[who[k][n]][k_adj[who[k][n]]][act_n_i]
                            a2_early = min(T_p[who[k][n]][k_adj[who[k][n]]][act_n_i] + s[act_n_i + A],
                                           a[act_n_i + 11 * A])
                            a2_late = max(T_p[who[k][n]][k_adj[who[k][n]]][act_n_i] + s[act_n_i + A],
                                           a[act_n_i + 11 * A])
                            a3_flex = flexible[who[k][l]][k_adj[who[k][l]]][act_j]
                            a3_early = T_p[who[k][l]][k_adj[who[k][l]]][act_j]
                            a3_late = a3_early + a3_flex
                            a4_flex = flexible[who[k][n]][k_adj[who[k][n]]][act_n_j]
                            a4_early = T_p[who[k][n]][k_adj[who[k][n]]][act_n_j]
                            a4_late = a4_early + a4_flex
                            if (i - 1) // A == 11 and (j - 1) // A == 1:
                                if (n_i - 1) // A == 11 and (n_j - 1) // A == 1:  # a1, a2, a3, a4 constrained
                                    if a2_early <= a1_early + t[i][n_i] and a1_late + t[i][n_i] <= a2_late and \
                                            a3_early <= a2_early + t[n_i][j] and a2_late + t[n_i][j] <= a3_late and \
                                            a4_early <= a3_early + t[j][n_j] and a3_late + t[j][n_j] <= a4_late and \
                                            t[i][n_i] + t[n_i][j] <= t[i][j] + a1_flex and \
                                            t[n_i][j] + t[j][n_j] <= t[n_i][n_j] + a2_flex and \
                                            t[i][n_i] + t[n_i][j] <= t[i][j] + a3_flex and \
                                            t[n_i][j] + t[j][n_j] <= t[n_i][n_j] + a4_flex:
                                        pl_cost[0] = beta_PAViv * (t[i][n_i] + 2 * t[n_i][j] + t[j][n_j]) + \
                                                     beta_PAVop * (d[i][n_i] + d[n_i][j] + d[j][n_j])
                                        pl_type[0] = "FIFO-PAV"
                                        new_mode[0] = "PAV"
                                    if a2_early <= a1_early + t[i][n_i] and a1_late + t[i][n_i] <= a2_late and \
                                            a3_early <= a4_early + t[n_j][j] and a4_late + t[n_j][j] <= a3_late and \
                                            t[i][n_i] + t[n_i][n_j] + t[n_j][j] <= t[i][j] + a1_flex and \
                                            t[i][n_i] + t[n_i][n_j] + t[n_j][j] <= t[i][j] + a3_flex:
                                        pl_cost[1] = beta_PAViv * (t[i][n_i] + 2 * t[n_i][n_j] + t[n_j][j]) + \
                                                     beta_PAVop * (d[i][n_i] + d[n_i][n_j] + d[n_j][j])
                                        pl_type[1] = "LIFO-PAV"
                                        new_mode[1] = "PAV"
                                elif (n_i - 1) // A == 11:  # a1, a2, a3 constrained
                                    if a2_early <= a1_early + t[i][n_i] and a1_late + t[i][n_i] <= a2_late and \
                                            a3_early <= a2_early + t[n_i][j] and a2_late + t[n_i][j] <= a3_late and \
                                            t[i][n_i] + t[n_i][j] <= t[i][j] + a1_flex and \
                                            t[n_i][j] + t[j][n_j] <= t[n_i][n_j] + a2_flex and \
                                            t[i][n_i] + t[n_i][j] <= t[i][j] + a3_flex:
                                        pl_cost[2] = beta_PAViv * (t[i][n_i] + 2 * t[n_i][j] + t[j][n_j]) + \
                                                     beta_PAVop * (d[i][n_i] + d[n_i][j] + d[j][n_j])
                                        pl_type[2] = "FIFO-PAV"
                                        new_mode[2] = "PAV"
                                    if a2_early <= a1_early + t[i][n_i] and a1_late + t[i][n_i] <= a2_late and \
                                            t[i][n_i] + t[n_i][n_j] + t[n_j][j] <= t[i][j] + a1_flex and \
                                            t[i][n_i] + t[n_i][n_j] + t[n_j][j] <= t[i][j] + a3_flex:
                                        pl_cost[3] = beta_PAViv * (t[i][n_i] + 2 * t[n_i][n_j] + t[n_j][j]) + \
                                                     beta_PAVop * (d[i][n_i] + d[n_i][n_j] + d[n_j][j])
                                        pl_type[3] = "LIFO-PAV"
                                        new_mode[3] = "PAV"
                                    stn = n_i + A
                                    trn_time = t_wlk[n_i][stn] + w_TRN + t_trn[n_i][stn]
                                    if a2_early + trn_time <= a1_early + t[i][stn] and \
                                            a1_late + t[i][stn] <= a2_late + trn_time and \
                                            a3_early <= a2_early + trn_time + t[n_i][j] and \
                                            a2_late + trn_time + t[stn][j] <= a3_late and \
                                            t[i][stn] + t[stn][j] <= t[i][j] + a1_flex and \
                                            trn_time + t[stn][j] + t[j][n_j] <= t[n_i][n_j] + a2_flex and \
                                            t[i][stn] + t[stn][j] <= t[i][j] + a3_flex:
                                        pl_cost[4] = beta_PAViv * (t[i][stn] + 2 * t[stn][j] + t[j][n_j]) + \
                                                     beta_PAVop * (d[i][stn] + d[stn][j] + d[j][n_j]) + \
                                                     beta_TRNwk * t_wlk[n_i][stn] + beta_TRNfr + beta_TRNwt * w_TRN + \
                                                     beta_TRNiv * t_trn[n_i][stn]
                                        pl_type[4] = "FIFO-P_T"
                                        new_mode[4] = "P_T"
                                    if a2_early + trn_time <= a1_early + t[i][stn] and \
                                            a1_late + t[i][stn] <= a2_late + trn_time and \
                                            t[i][stn] + t[stn][n_j] + t[n_j][j] <= t[i][j] + a1_flex and \
                                            trn_time + t[stn][n_j] <= t[n_i][n_j] + a2_flex and \
                                            t[i][stn] + t[stn][n_j] + t[n_j][j] <= t[i][j] + a3_flex:
                                        pl_cost[5] = beta_PAViv * (t[i][stn] + 2 * t[stn][n_j] + t[n_j][j]) + \
                                                     beta_PAVop * (d[i][stn] + d[stn][n_j] + d[n_j][j]) + \
                                                     beta_TRNwk * t_wlk[n_i][stn] + beta_TRNfr + beta_TRNwt * w_TRN + \
                                                     beta_TRNiv * t_trn[n_i][stn]
                                        pl_type[5] = "LIFO-P_T"
                                        new_mode[5] = "P_T"
                                elif (n_j - 1) // A == 1:  # a1, a3, a4 constrained
                                    if a4_early <= a3_early + t[j][n_j] and a3_late + t[j][n_j] <= a4_late and \
                                            t[i][n_i] + t[n_i][j] <= t[i][j] + a1_flex and \
                                            t[i][n_i] + t[n_i][j] <= t[i][j] + a3_flex and \
                                            t[n_i][j] + t[j][n_j] <= t[n_i][n_j] + a4_flex:
                                        pl_cost[6] = beta_PAViv * (t[i][n_i] + 2 * t[n_i][j] + t[j][n_j]) + \
                                                     beta_PAVop * (d[i][n_i] + d[n_i][j] + d[j][n_j])
                                        pl_type[6] = "FIFO-PAV"
                                        new_mode[6] = "PAV"
                                    if a3_early <= a4_early + t[n_j][j] and a4_late + t[n_j][j] <= a3_late and \
                                            t[i][n_i] + t[n_i][n_j] + t[n_j][j] <= t[i][j] + a1_flex and \
                                            t[i][n_i] + t[n_i][n_j] + t[n_j][j] <= t[i][j] + a3_flex:
                                        pl_cost[7] = beta_PAViv * (t[i][n_i] + 2 * t[n_i][n_j] + t[n_j][j]) + \
                                                     beta_PAVop * (d[i][n_i] + d[n_i][n_j] + d[n_j][j])
                                        pl_type[7] = "LIFO-PAV"
                                        new_mode[7] = "PAV"
                                    stn = n_j + A
                                    trn_time = t_wlk[stn][n_j] + w_TRN + t_trn[stn][n_j]
                                    if a4_early <= a3_early + t[n_j][j] and a3_late + t[n_j][j] <= a4_late and \
                                            t[i][n_i] + t[n_i][j] <= t[i][j] + a1_flex and \
                                            t[i][n_i] + t[n_i][j] <= t[i][j] + a3_flex and \
                                            t[n_i][j] + t[j][stn] + trn_time <= t[n_i][n_j] + a4_flex:
                                        pl_cost[8] = beta_PAViv * (t[i][n_i] + 2 * t[n_i][j] + t[j][stn]) + \
                                                     beta_PAVop * (d[i][n_i] + d[n_i][j] + d[j][stn]) + \
                                                     beta_TRNwk * t_wlk[stn][n_j] + beta_TRNfr + beta_TRNwt * w_TRN + \
                                                     beta_TRNiv * t_trn[stn][n_j]
                                        pl_type[8] = "FIFO-P_T"
                                        new_mode[8] = "P_T"
                                    if a3_early <= a4_early + t[n_j][j] and a4_late + t[n_j][j] <= a3_late and \
                                            t[i][n_i] + t[n_i][stn] + t[stn][j] <= t[i][j] + a1_flex and \
                                            t[i][n_i] + t[n_i][stn] + t[stn][j] <= t[i][j] + a3_flex and \
                                            t[n_i][stn] + trn_time <= t[n_i][n_j] + a4_flex:
                                        pl_cost[9] = beta_PAViv * (t[i][n_i] + 2 * t[n_i][stn] + t[stn][j]) + \
                                                     beta_PAVop * (d[i][n_i] + d[n_i][stn] + d[stn][j]) + \
                                                     beta_TRNwk * t_wlk[stn][n_j] + beta_TRNfr + beta_TRNwt * w_TRN + \
                                                     beta_TRNiv * t_trn[stn][n_j]
                                        pl_type[9] = "LIFO-P_T"
                                        new_mode[9] = "P_T"
                            elif (i - 1) // A == 11:
                                if (n_i - 1) // A == 11 and (n_j - 1) // A == 1:  # a1, a2, a4 constrained
                                    if a2_early <= a1_early + t[i][n_i] and a1_late + t[i][n_i] <= a2_late and \
                                            t[i][n_i] + t[n_i][j] <= t[i][j] + a1_flex and \
                                            t[n_i][j] + t[j][n_j] <= t[n_i][n_j] + a2_flex and \
                                            t[n_i][j] + t[j][n_j] <= t[n_i][n_j] + a4_flex:
                                        pl_cost[10] = beta_PAViv * (t[i][n_i] + 2 * t[n_i][j] + t[j][n_j]) + \
                                                      beta_PAVop * (d[i][n_i] + d[n_i][j] + d[j][n_j])
                                        pl_type[10] = "FIFO-PAV"
                                        new_mode[10] = "PAV"
                                    if a2_early <= a1_early + t[i][n_i] and a1_late + t[i][n_i] <= a2_late and \
                                            t[i][n_i] + t[n_i][n_j] + t[n_j][j] <= t[i][j] + a1_flex:
                                        pl_cost[11] = beta_PAViv * (t[i][n_i] + 2 * t[n_i][n_j] + t[n_j][j]) + \
                                                      beta_PAVop * (d[i][n_i] + d[n_i][n_j] + d[n_j][j])
                                        pl_type[11] = "LIFO-PAV"
                                        new_mode[11] = "PAV"
                                elif (n_i - 1) // A == 11:  # a1, a2 constrained
                                    if a2_early <= a1_early + t[i][n_i] and a1_late + t[i][n_i] <= a2_late and \
                                            t[i][n_i] + t[n_i][j] <= t[i][j] + a1_flex and \
                                            t[n_i][j] + t[j][n_j] <= t[n_i][n_j] + a2_flex:
                                        pl_cost[12] = beta_PAViv * (t[i][n_i] + 2 * t[n_i][j] + t[j][n_j]) + \
                                                      beta_PAVop * (d[i][n_i] + d[n_i][j] + d[j][n_j])
                                        pl_type[12] = "FIFO-PAV"
                                        new_mode[12] = "PAV"
                                    if a2_early <= a1_early + t[i][n_i] and a1_late + t[i][n_i] <= a2_late and \
                                            t[i][n_i] + t[n_i][n_j] + t[n_j][j] <= t[i][j] + a1_flex:
                                        pl_cost[13] = beta_PAViv * (t[i][n_i] + 2 * t[n_i][n_j] + t[n_j][j]) + \
                                                      beta_PAVop * (d[i][n_i] + d[n_i][n_j] + d[n_j][j])
                                        pl_type[13] = "LIFO-PAV"
                                        new_mode[13] = "PAV"
                                    stn = n_i + A
                                    trn_time = t_wlk[n_i][stn] + w_TRN + t_trn[n_i][stn]
                                    if a2_early + trn_time <= a1_early + t[i][stn] and \
                                            a1_late + t[i][stn] <= a2_late + trn_time and \
                                            t[i][stn] + t[stn][j] <= t[i][j] + a1_flex and \
                                            trn_time + t[stn][j] + t[j][n_j] <= t[n_i][n_j] + a2_flex:
                                        pl_cost[14] = beta_PAViv * (t[i][stn] + 2 * t[stn][j] + t[j][n_j]) + \
                                                      beta_PAVop * (d[i][stn] + d[stn][j] + d[j][n_j]) + \
                                                      beta_TRNwk * t_wlk[n_i][stn] + beta_TRNfr + beta_TRNwt * w_TRN + \
                                                      beta_TRNiv * t_trn[n_i][stn]
                                        pl_type[14] = "FIFO-P_T"
                                        new_mode[14] = "P_T"
                                    if a2_early + trn_time <= a1_early + t[i][stn] and \
                                            a1_late + t[i][stn] <= a2_late + trn_time and \
                                            t[i][stn] + t[stn][n_j] + t[n_j][j] <= t[i][j] + a1_flex and \
                                            trn_time + t[stn][n_j] <= t[n_i][n_j] + a2_flex:
                                        pl_cost[15] = beta_PAViv * (t[i][stn] + 2 * t[stn][n_j] + t[n_j][j]) + \
                                                      beta_PAVop * (d[i][stn] + d[stn][n_j] + d[n_j][j]) + \
                                                      beta_TRNwk * t_wlk[n_i][stn] + beta_TRNfr + beta_TRNwt * w_TRN + \
                                                      beta_TRNiv * t_trn[n_i][stn]
                                        pl_type[15] = "LIFO-P_T"
                                        new_mode[15] = "P_T"
                                elif (n_j - 1) // A == 1:  # a1, a4 constrained
                                    if a4_early <= a1_early + t[i][n_i] + t[n_i][j] + t[j][n_j] and \
                                            a1_late + t[i][n_i] + t[n_i][j] + t[j][n_j] <= a4_late and \
                                            t[i][n_i] + t[n_i][j] <= t[i][j] + a1_flex and \
                                            t[n_i][j] + t[j][n_j] <= t[n_i][n_j] + a4_flex:
                                        pl_cost[16] = beta_PAViv * (t[i][n_i] + 2 * t[n_i][j] + t[j][n_j]) + \
                                                      beta_PAVop * (d[i][n_i] + d[n_i][j] + d[j][n_j])
                                        pl_type[16] = "FIFO-PAV"
                                        new_mode[16] = "PAV"
                                    if a4_early <= a1_early + t[i][n_i] + t[n_i][n_j] and \
                                            a1_late + t[i][n_i] + t[n_i][n_j] <= a4_late and \
                                            t[i][n_i] + t[n_i][n_j] + t[n_j][j] <= t[i][j] + a1_flex:
                                        pl_cost[17] = beta_PAViv * (t[i][n_i] + 2 * t[n_i][n_j] + t[n_j][j]) + \
                                                      beta_PAVop * (d[i][n_i] + d[n_i][n_j] + d[n_j][j])
                                        pl_type[17] = "LIFO-PAV"
                                        new_mode[17] = "PAV"
                                    stn = n_j + A
                                    trn_time = t_wlk[stn][n_j] + w_TRN + t_trn[stn][n_j]
                                    if a4_early <= a1_early + t[i][n_i] + t[n_i][j] + t[j][stn] + trn_time and \
                                            a1_late + t[i][n_i] + t[n_i][j] + t[j][stn] + trn_time <= a4_late and \
                                            t[i][n_i] + t[n_i][j] <= t[i][j] + a1_flex and \
                                            t[n_i][j] + t[j][stn] + trn_time <= t[n_i][n_j] + a4_flex:
                                        pl_cost[18] = beta_PAViv * (t[i][n_i] + 2 * t[n_i][j] + t[j][stn]) + \
                                                      beta_PAVop * (d[i][n_i] + d[n_i][j] + d[j][stn]) + \
                                                      beta_TRNwk * t_wlk[stn][n_j] + beta_TRNfr + beta_TRNwt * w_TRN + \
                                                      beta_TRNiv * t_trn[stn][n_j]
                                        pl_type[18] = "FIFO-P_T"
                                        new_mode[18] = "P_T"
                                    if a4_early <= a1_early + t[i][n_i] + t[n_i][stn] + trn_time and \
                                            a1_late + t[i][n_i] + t[n_i][stn] + trn_time <= a4_late and \
                                            t[i][n_i] + t[n_i][stn] + t[stn][j] <= t[i][j] + a1_flex and \
                                            t[n_i][stn] + trn_time <= t[n_i][n_j] + a4_flex:
                                        pl_cost[19] = beta_PAViv * (t[i][n_i] + 2 * t[n_i][stn] + t[stn][j]) + \
                                                      beta_PAVop * (d[i][n_i] + d[n_i][stn] + d[stn][j]) + \
                                                      beta_TRNwk * t_wlk[stn][n_j] + beta_TRNfr + beta_TRNwt * w_TRN + \
                                                      beta_TRNiv * t_trn[stn][n_j]
                                        pl_type[19] = "LIFO-P_T"
                                        new_mode[19] = "P_T"
                            elif (j - 1) // A == 1:
                                if (n_i - 1) // A == 11 and (n_j - 1) // A == 1:  # a2, a3, a4 constrained
                                    if a3_early <= a2_early + t[n_i][j] and a2_late + t[n_i][j] <= a3_late and \
                                            a4_early <= a3_early + t[j][n_j] and a3_late + t[j][n_j] <= a4_late and \
                                            t[n_i][j] + t[j][n_j] <= t[n_i][n_j] + a2_flex and \
                                            t[i][n_i] + t[n_i][j] <= t[i][j] + a3_flex and \
                                            t[n_i][j] + t[j][n_j] <= t[n_i][n_j] + a4_flex:
                                        pl_cost[20] = beta_PAViv * (t[i][n_i] + 2 * t[n_i][j] + t[j][n_j]) + \
                                                      beta_PAVop * (d[i][n_i] + d[n_i][j] + d[j][n_j])
                                        pl_type[20] = "FIFO-PAV"
                                        new_mode[20] = "PAV"
                                    if a3_early <= a4_early + t[n_j][j] and a4_late + t[n_j][j] <= a3_late and \
                                            t[i][n_i] + t[n_i][n_j] + t[n_j][j] <= t[i][j] + a3_flex:
                                        pl_cost[21] = beta_PAViv * (t[i][n_i] + 2 * t[n_i][n_j] + t[n_j][j]) + \
                                                     beta_PAVop * (d[i][n_i] + d[n_i][n_j] + d[n_j][j])
                                        pl_type[21] = "LIFO-PAV"
                                        new_mode[21] = "PAV"
                                elif (n_i - 1) // A == 11:  # a2, a3 constrained
                                    if a3_early <= a2_early + t[n_i][j] and a2_late + t[n_i][j] <= a3_late and \
                                            t[n_i][j] + t[j][n_j] <= t[n_i][n_j] + a2_flex and \
                                            t[i][n_i] + t[n_i][j] <= t[i][j] + a3_flex:
                                        pl_cost[22] = beta_PAViv * (t[i][n_i] + 2 * t[n_i][j] + t[j][n_j]) + \
                                                      beta_PAVop * (d[i][n_i] + d[n_i][j] + d[j][n_j])
                                        pl_type[22] = "FIFO-PAV"
                                        new_mode[22] = "PAV"
                                    if a3_early <= a4_early + t[n_j][j] and a4_late + t[n_j][j] <= a3_late and \
                                            t[i][n_i] + t[n_i][n_j] + t[n_j][j] <= t[i][j] + a3_flex:
                                        pl_cost[23] = beta_PAViv * (t[i][n_i] + 2 * t[n_i][n_j] + t[n_j][j]) + \
                                                      beta_PAVop * (d[i][n_i] + d[n_i][n_j] + d[n_j][j])
                                        pl_type[23] = "LIFO-PAV"
                                        new_mode[23] = "PAV"
                                    stn = n_i + A
                                    trn_time = t_wlk[n_i][stn] + w_TRN + t_trn[n_i][stn]
                                    if a3_early <= a2_early + trn_time + t[n_i][j] and \
                                            a2_late + trn_time + t[stn][j] <= a3_late and \
                                            trn_time + t[stn][j] + t[j][n_j] <= t[n_i][n_j] + a2_flex and \
                                            t[i][stn] + t[stn][j] <= t[i][j] + a3_flex:
                                        pl_cost[24] = beta_PAViv * (t[i][stn] + 2 * t[stn][j] + t[j][n_j]) + \
                                                      beta_PAVop * (d[i][stn] + d[stn][j] + d[j][n_j]) + \
                                                      beta_TRNwk * t_wlk[n_i][stn] + beta_TRNfr + beta_TRNwt * w_TRN + \
                                                      beta_TRNiv * t_trn[n_i][stn]
                                        pl_type[24] = "FIFO-P_T"
                                        new_mode[24] = "P_T"
                                    if a2_early + trn_time + t[stn][n_j] + t[n_j][j] <= a3_early and \
                                            a3_late <= a2_late + trn_time + t[stn][n_j] + t[n_j][j] and \
                                            trn_time + t[stn][n_j] <= t[n_i][n_j] + a2_flex and \
                                            t[i][stn] + t[stn][n_j] + t[n_j][j] <= t[i][j] + a3_flex:
                                        pl_cost[25] = beta_PAViv * (t[i][stn] + 2 * t[stn][n_j] + t[n_j][j]) + \
                                                      beta_PAVop * (d[i][stn] + d[stn][n_j] + d[n_j][j]) + \
                                                      beta_TRNwk * t_wlk[n_i][stn] + beta_TRNfr + beta_TRNwt * w_TRN + \
                                                      beta_TRNiv * t_trn[n_i][stn]
                                        pl_type[25] = "LIFO-P_T"
                                        new_mode[25] = "P_T"
                                elif (n_j - 1) // A == 1:  # a3, a4 constrained
                                    if a4_early <= a3_early + t[j][n_j] and a3_late + t[j][n_j] <= a4_late and \
                                            t[i][n_i] + t[n_i][j] <= t[i][j] + a3_flex and \
                                            t[n_i][j] + t[j][n_j] <= t[n_i][n_j] + a4_flex:
                                        pl_cost[26] = beta_PAViv * (t[i][n_i] + 2 * t[n_i][j] + t[j][n_j]) + \
                                                      beta_PAVop * (d[i][n_i] + d[n_i][j] + d[j][n_j])
                                        pl_type[26] = "FIFO-PAV"
                                        new_mode[26] = "PAV"
                                    if a3_early <= a4_early + t[n_j][j] and a4_late + t[n_j][j] <= a3_late and \
                                            t[i][n_i] + t[n_i][n_j] + t[n_j][j] <= t[i][j] + a3_flex:
                                        pl_cost[27] = beta_PAViv * (t[i][n_i] + 2 * t[n_i][n_j] + t[n_j][j]) + \
                                                      beta_PAVop * (d[i][n_i] + d[n_i][n_j] + d[n_j][j])
                                        pl_type[27] = "LIFO-PAV"
                                        new_mode[27] = "PAV"
                                    stn = n_j + A
                                    trn_time = t_wlk[stn][n_j] + w_TRN + t_trn[stn][n_j]
                                    if a4_early <= a3_early + t[n_j][j] and a3_late + t[n_j][j] <= a4_late and \
                                            t[i][n_i] + t[n_i][j] <= t[i][j] + a3_flex and \
                                            t[n_i][j] + t[j][stn] + trn_time <= t[n_i][n_j] + a4_flex:
                                        pl_cost[28] = beta_PAViv * (t[i][n_i] + 2 * t[n_i][j] + t[j][stn]) + \
                                                      beta_PAVop * (d[i][n_i] + d[n_i][j] + d[j][stn]) + \
                                                      beta_TRNwk * t_wlk[stn][n_j] + beta_TRNfr + beta_TRNwt * w_TRN + \
                                                      beta_TRNiv * t_trn[stn][n_j]
                                        pl_type[28] = "FIFO-P_T"
                                        new_mode[28] = "P_T"
                                    if a3_early <= a4_early + t[n_j][j] and a4_late + t[n_j][j] <= a3_late and \
                                            t[i][n_i] + t[n_i][stn] + t[stn][j] <= t[i][j] + a3_flex and \
                                            t[n_i][stn] + trn_time <= t[n_i][n_j] + a4_flex:
                                        pl_cost[29] = beta_PAViv * (t[i][n_i] + 2 * t[n_i][stn] + t[stn][j]) + \
                                                      beta_PAVop * (d[i][n_i] + d[n_i][stn] + d[stn][j]) + \
                                                      beta_TRNwk * t_wlk[stn][n_j] + beta_TRNfr + beta_TRNwt * w_TRN + \
                                                      beta_TRNiv * t_trn[stn][n_j]
                                        pl_type[29] = "LIFO-P_T"
                                        new_mode[29] = "P_T"
                            pl_costs = [pl_cost[0], pl_cost[1], pl_cost[2], pl_cost[3], pl_cost[4], pl_cost[5],
                                        pl_cost[6], pl_cost[7], pl_cost[8], pl_cost[9], pl_cost[10], pl_cost[11],
                                        pl_cost[12], pl_cost[13], pl_cost[14], pl_cost[15], pl_cost[16], pl_cost[17],
                                        pl_cost[18], pl_cost[19], pl_cost[20], pl_cost[21], pl_cost[22], pl_cost[23],
                                        pl_cost[24], pl_cost[25], pl_cost[26], pl_cost[27], pl_cost[28], pl_cost[29],
                                        prev_cost]
                            for code in range(len(pl_costs) - 1):
                                if min(pl_costs) == pl_cost[code]:
                                    pl_code = code
                                    pl_type_selected = pl_type[code]
                                    new_mode_selected = new_mode[code]
                            if pl_code in [0, 1, 2, 3, 6, 7, 10, 11, 12, 13, 16, 17, 20, 21, 22, 23, 26, 27]:
                                if mode_TRV[k][n] == "PAV" or mode_TRV[k][n] == "P_T":
                                    T_v[k][n_veh].remove(T_v[k][n_veh][L_v[k][n_veh].index([n_i, n_j])])
                                    L_v[k][n_veh].remove(L_v[k][n_veh][L_v[k][n_veh].index([n_i, n_j])])
                                if pl_code in [0, 2, 6, 10, 12, 16, 20, 22, 26]:
                                    time_diff = t[i][n_i] + t[n_i][j] + t[j][n_j] - t[i][j]
                                    old_depart = T_v[k][l_veh][L_v[k][l_veh].index([i, j])][0]
                                    old_arrival = T_v[k][l_veh][L_v[k][l_veh].index([i, j])][1]
                                    T_v[k][l_veh].append([old_depart, old_arrival + time_diff])
                                    T_v[k][l_veh].remove(T_v[k][l_veh][L_v[k][l_veh].index([i, j])])
                                    L_v[k][l_veh].remove(L_v[k][l_veh][L_v[k][l_veh].index([i, j])])
                                    L_v[k][l_veh].append([i, n_j])
                                elif pl_code in [1, 3, 7, 11, 13, 17, 21, 23, 27]:
                                    time_diff = t[i][n_i] + t[n_i][n_j] + t[n_j][j] - t[i][j]
                                    T_v[k][l_veh][L_v[k][l_veh].index([i, j])][1] += time_diff
                                if new_mode != "":
                                    mode_TRV[k][n] = new_mode_selected
                                total_cost[k] += pl_cost[pl_code]
                                total_cost[k] -= prev_cost_ob
                                pooled_TRV[k].append([i, j, mode_TRV[k][l], n_i, n_j, mode_TRV[k][n],
                                                      pl_code, pl_type_selected])
                                is_pooled[k].append([i, j])
                                is_pooled[k].append([n_i, n_j])
                            elif pl_code in [4, 5, 8, 9, 14, 15, 18, 19, 24, 25, 28, 29]:
                                if mode_TRV[k][n] == "PAV" or mode_TRV[k][n] == "P_T":
                                    T_v[k][n_veh].remove(T_v[k][n_veh][L_v[k][n_veh].index([n_i, n_j])])
                                    L_v[k][n_veh].remove(L_v[k][n_veh][L_v[k][n_veh].index([n_i, n_j])])
                                if pl_code in [4, 14, 24]:
                                    stn = n_i + A
                                    time_diff = t[i][stn] + t[stn][j] + t[j][n_j] - t[i][j]
                                    old_depart = T_v[k][l_veh][L_v[k][l_veh].index([i, j])][0]
                                    old_arrival = T_v[k][l_veh][L_v[k][l_veh].index([i, j])][1]
                                    T_v[k][l_veh].append([old_depart, old_arrival + time_diff])
                                    T_v[k][l_veh].remove(T_v[k][l_veh][L_v[k][l_veh].index([i, j])])
                                    L_v[k][l_veh].remove(L_v[k][l_veh][L_v[k][l_veh].index([i, j])])
                                    L_v[k][l_veh].append([i, n_j])
                                elif pl_code in [5, 15, 25]:
                                    stn = n_i + A
                                    time_diff = t[i][stn] + t[stn][n_j] + t[n_j][j] - t[i][j]
                                    T_v[k][l_veh][L_v[k][l_veh].index([i, j])][1] += time_diff
                                elif pl_code in [8, 18, 28]:
                                    stn = n_j + A
                                    time_diff = t[i][n_i] + t[n_i][j] + t[j][stn] - t[i][j]
                                    old_depart = T_v[k][l_veh][L_v[k][l_veh].index([i, j])][0]
                                    old_arrival = T_v[k][l_veh][L_v[k][l_veh].index([i, j])][1]
                                    T_v[k][l_veh].append([old_depart, old_arrival + time_diff])
                                    T_v[k][l_veh].remove(T_v[k][l_veh][L_v[k][l_veh].index([i, j])])
                                    L_v[k][l_veh].remove(L_v[k][l_veh][L_v[k][l_veh].index([i, j])])
                                    L_v[k][l_veh].append([i, stn])
                                elif pl_code in [9, 19, 29]:
                                    stn = n_j + A
                                    time_diff = t[i][n_i] + t[n_i][stn] + t[stn][j] - t[i][j]
                                    T_v[k][l_veh][L_v[k][l_veh].index([i, j])][1] += time_diff
                                if not new_mode == "":
                                    mode_TRV[k][n] = new_mode_selected
                                total_cost[k] += pl_cost[pl_code]
                                total_cost[k] -= prev_cost_ob
                                pooled_TRV[k].append([i, j, mode_TRV[k][l], n_i, n_j, mode_TRV[k][n],
                                                      pl_code, pl_type_selected])
                                is_pooled[k].append([i, j])
                                is_pooled[k].append([n_i, n_j])
            # Deadheading decision
            dh_cost[k] = OrderedDict()
            who_wait = OrderedDict()
            for v in V:
                dh_cost[k][v] = 0
                if len(L_v[k][v]) >= 1:
                    sorted_indices[k][v] = sorted(range(len(T_v[k][v])), key=lambda i: T_v[k][v][i][0])
                    sorted_l_time[v] = [lth[k][v][i] for i in sorted_indices[k][v]]
                    dh_cost_first = 0
                    dh_cost_mid = 0
                    dh_cost_last = 0
                    for l in range(len(sorted_indices[k][v])):
                        if l == 0 and L[L_v[k][v][sorted_indices[k][v][l]][0]] != HL:  # first PAV trip
                            dh_cost_first = beta_PAVop * d[1][L_v[k][v][sorted_indices[k][v][l]][0]]
                        else:
                            if len(T_v[k][v]) > 1 and l != 0:
                                i = L_v[k][v][sorted_indices[k][v][l - 1]][1]
                                j = L_v[k][v][sorted_indices[k][v][l]][0]
                                dh_time = T_v[k][v][sorted_indices[k][v][l]][0] - \
                                          T_v[k][v][sorted_indices[k][v][l - 1]][1]
                                TRV_index_i = next((q for q, sublist in enumerate(act_TRV_com[k])
                                                    if sublist[1] == i), None)
                                TRV_index_j = next((q for q, sublist in enumerate(act_TRV_com[k])
                                                    if sublist[0] == j), None)
                                jumped = TRV_index_j - TRV_index_i
                                if jumped > 1:
                                    for r in range(1, jumped + 1):
                                        if mode_TRV[k][TRV_index_i + r] == "Trn":
                                            mid_TRV_i = act_TRV_com[k][TRV_index_i + r][0]
                                            mid_TRV_j = act_TRV_com[k][TRV_index_i + r][1]
                                            dh_time += (t_wlk[mid_TRV_i][mid_TRV_j] + t_trn[mid_TRV_i][mid_TRV_j] +
                                                        w_TRN - t[mid_TRV_i][mid_TRV_j])
                                if ((i - 1) // A == 1 or (i - 1) // A == 14) and \
                                        ((j - 1) // A == 0 or (j - 1) // A == 11):  # no onboard passengers
                                    next_TRV_index = next((q for q, sublist in enumerate(act_TRV_com[k])
                                                           if sublist[0] == j), None)
                                    if dh_time <= t[i][j] + 5 and mode_TRV[k][next_TRV_index] == "PAV":
                                        dh_cost_mid += beta_PAVop * d[i][j]
                                        T_v[k][v][sorted_indices[k][v][l]][0] += (t[i][j] - dh_time)
                                        T_v[k][v][sorted_indices[k][v][l]][1] += (t[i][j] - dh_time)
                                    else:
                                        if j % A == 0:
                                            act_p = A
                                        else:
                                            act_p = j % A
                                        dh_cost_pk_n = beta_PAVop * (d[i][act_p + 4 * A] + d[act_p + 4 * A][j]) + \
                                                       pkf[act_p + 4 * A] / 60 * \
                                                       (dh_time - (t[i][act_p + 4 * A] + t[act_p + 4 * A][j]))
                                        dh_cost_pk_r = beta_PAVop * (d[i][act_p + 7 * A] + d[act_p + 7 * A][j]) + \
                                                       pkf[act_p + 7 * A] / 60 * \
                                                       (dh_time - (t[i][act_p + 7 * A] + t[act_p + 7 * A][j]))
                                        dh_cost_pk_h = beta_PAVop * (d[i][act_p + 10 * A] + d[act_p + 10 * A][j])
                                        if min(dh_cost_pk_n, dh_cost_pk_r, dh_cost_pk_h) == dh_cost_pk_n:
                                            L_pk[k].append(act_p + 4 * A)
                                            dh_cost_mid += dh_cost_pk_n
                                        elif min(dh_cost_pk_n, dh_cost_pk_r, dh_cost_pk_h) == dh_cost_pk_r:
                                            L_pk[k].append(act_p + 7 * A)
                                            dh_cost_mid += dh_cost_pk_r
                                        else:
                                            # if (act_p + 10 * A in L_pk[k]) or \
                                            #         (t[i][act_p + 10 * A] + t[act_p + 10 * A][j] > dh_time):
                                            #     if dh_cost_pk_n <= dh_cost_pk_r:
                                            #         L_pk[k].append(act_p + 4 * A)
                                            #         dh_cost_mid += dh_cost_pk_n
                                            #     else:
                                            #         L_pk[k].append(act_p + 7 * A)
                                            #         dh_cost_mid += dh_cost_pk_r
                                            # else:
                                            L_pk[k].append(act_p + 10 * A)
                                            dh_cost_mid += dh_cost_pk_h
                                if (j - 1) // A == 11:
                                    if j % A == 0:
                                        act_p = A
                                    else:
                                        act_p = j % A
                                    for p in P:
                                        if act_p in theta[p]:
                                            who_wait[j] = p
                                    if i % A == 0:
                                        act_p_i = A
                                    else:
                                        act_p_i = i % A
                                    for p in P:
                                        if act_p_i in theta[p]:
                                            who_wait[i] = p
                                    flex_j = flexible[who_wait[j]][k_adj[who_wait[j]]][act_p]
                                    if s[act_p + A] != 5 and who_wait[i] != who_wait[j]:
                                        if T_v[k][v][sorted_indices[k][v][l]][0] - \
                                                (a[act_p + A] + s[act_p + A]) > flex_j:
                                            wait_pav[k] += T_v[k][v][sorted_indices[k][v][l]][0] - \
                                                           (a[act_p + A] + s[act_p + A]) - flex_j
                        if L[L_v[k][v][sorted_indices[k][v][-1]][1]] != HL:
                            dh_cost_last = beta_PAVop * d[L_v[k][v][sorted_indices[k][v][-1]][1]][15 * A]
                    dh_cost[k][v] = dh_cost_first + dh_cost_mid + dh_cost_last
                    total_cost[k] += dh_cost[k][v]
            total_cost[k] += (beta_PAVwt * wait_pav[k])
            for v in V:
                route_PAV[k][v] = []
                for l in range(len(L_v[k][v])):
                    route_PAV[k][v].append(["Loc:", L_v[k][v][sorted_indices[k][v][l]],
                                            "Time:", T_v[k][v][sorted_indices[k][v][l]]])
        # No PAV households
        else:
            for l in range(len(act_TRV_com[k])):
                mode_TRV[k].append([])
                i = act_TRV_com[k][l][0]
                j = act_TRV_com[k][l][1]
                if sorted_modes[k][l][0] == "PAV" or sorted_modes[k][l][0] == "P_T":
                    if sorted_modes[k][l][1] == "P_T" or sorted_modes[k][l][1] == "PAV":
                        mode_TRV[k][l] = sorted_modes[k][l][2]
                    else:
                        mode_TRV[k][l] = sorted_modes[k][l][1]
                else:
                    mode_TRV[k][l] = sorted_modes[k][l][0]
                total_cost[k] = total_cost[k] + cost_TRV[k][l][mode_TRV[k][l]]

    # Step 3: Routing and scheduling choice ========================================================================== #
    ih_incumbent = 9999
    best_k = 0
    for k in range(len(act_TRV_com)):
        if total_cost[k] < ih_incumbent:
            ih_incumbent = total_cost[k]
            best_k = k
    print("")
    runtime = round(time.time() - timer[h], 2)
    print("Runtime =", runtime, "seconds")
    print("")

    # Total results
    n_PAV = n_SAV = n_Trn = n_P_T = n_S_T = 0
    for m in mode_TRV[best_k]:
        if m == "PAV":
            n_PAV += 1
        elif m == "SAV":
            n_SAV += 1
        elif m == "Trn":
            n_Trn += 1
        elif m == "P_T":
            n_P_T += 1
        elif m == "S_T":
            n_S_T += 1
    dh_cost_total = 0
    for v in V:
        dh_cost_total += dh_cost[best_k][v]

    print("* RESULT")
    print("Total onboard trips:", len(mode_TRV[best_k]))
    print("- PAV:", n_PAV, "/ SAV:", n_SAV, "/ Trn:", n_Trn, "/ P_T:", n_P_T, "/ S_T:", n_S_T)
    if len(V) != 0:
        print("- Pooling:", pooled_TRV[best_k], "/ Parking:", L_pk[best_k])
    print("Best routes")
    print("- Route of person trips:", act_TRV_com[best_k])
    for v in V:
        print("- Route of PAV", str(v) + ":", route_PAV[best_k][v])
    if len(V) != 0:
        print("- PAV deadheading cost:", dh_cost_total, "USD")
    print("Total cost:", total_cost[best_k], "USD")
    print("")
    record_HH = [{'hhid': hhid, 'hh_size': len(P), 'activities': A, 'PAVs': len(V),
                  'runtime': round(time.time() - timer[h], 2), 'obj_value': total_cost[best_k]}]
    df_ih_result = pd.DataFrame(record_HH)
    df_ih_result.to_csv('result_ih.csv', mode='a', index=False, header=False)
    if hhid == 350:
        print("stop")

# ==================================================================================================================== #
# TERMINATION
# ==================================================================================================================== #
print("")
print("HOUSEHOLD AV TRAVEL PLANNING IS COMPLETED!")

print("Total runtime =", round(time.time() - start_time, 2), "seconds")
