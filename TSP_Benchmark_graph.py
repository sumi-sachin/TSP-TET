import math
import random
import numpy as np
import networkx as nx
from networkx.algorithms import approximation as approx
import csv
import os
import shutil

INF = float('inf')

# ============================================================
# 1. READ TSPLIB EUC_2D COORDINATES (a280)
# ============================================================
def load_tsplib_coords(filepath):
    coords = {}
    with open(filepath, "r") as f:
        start = False
        for line in f:
            line = line.strip()
            if line == "NODE_COORD_SECTION":
                start = True
                continue
            if line == "EOF":
                break
            if start:
                parts = line.split()
                idx = int(parts[0])
                x = float(parts[1])
                y = float(parts[2])
                coords[idx] = (x, y)
    return coords


# ============================================================
# 2. BUILD WEIGHTED ERDŐS–RÉNYI GRAPH WITH EUCLIDEAN WEIGHTS
# ============================================================
def build_er_graph_from_coords(coords, p=0.25, seed=2025):
    random.seed(seed)
    G = nx.Graph()

    # add nodes with coordinates
    for v, (x, y) in coords.items():
        G.add_node(v, pos=(x, y))

    nodes = list(coords.keys())
    sedge = 0
    cnt = 0
    edge_weights = []
    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            if random.random() <= p:
                u, v = nodes[i], nodes[j]
                x1, y1 = coords[u]
                x2, y2 = coords[v]
                dist = math.hypot(x1 - x2, y1 - y2)
                G.add_edge(u, v, weight=dist)
                #sedge = sedge + dist
                edge_weights.append(dist)
                #cnt = cnt + 1

    #muedge = sedge / cnt
    mean_edge = np.mean(edge_weights)
    sigma_edge = np.std(edge_weights)
    print (f"mu egde weight = {mean_edge:.2f}")
    print (f"sigma egde weight = {sigma_edge:.2f}")
    if not nx.is_connected(G):
        return None
    return G


# ============================================================
# 3. METRIC CLOSURE
# ============================================================
def compute_metric_closure_distances(G):
    nodes = list(G.nodes())
    spl = dict(nx.all_pairs_dijkstra_path_length(G, weight="weight"))
    n = len(nodes)

    D = [[INF] * n for _ in range(n)]
    idx = {nodes[i]: i for i in range(n)}

    for u in spl:
        for v, d in spl[u].items():
            D[idx[u]][idx[v]] = d

    for i in range(n):
        D[i][i] = 0.0

    D_map = {nodes[i]: {nodes[j]: D[i][j] for j in range(n)} for i in range(n)}
    return nodes, D, D_map


# ============================================================
# 4. HAMILTONIAN CYCLE (CHRISTOFIDES)
# ============================================================
def compute_hamiltonian_cycle_from_metric(D, nodes):
    K = nx.Graph()
    n = len(nodes)

    for i in range(n):
        for j in range(i + 1, n):
            K.add_edge(nodes[i], nodes[j], weight=D[i][j])

    cyc = approx.traveling_salesman_problem(
        K, weight="weight", cycle=True, method=approx.christofides
    )

    if cyc[0] == cyc[-1]:
        cyc = cyc[:-1]
    return cyc


# ============================================================
# 5. HEURISTIC TSP-TET
# ============================================================
# ---------------- heuristic TSP-TET on the ordering ----------------
def tsp_tet_heuristic_from_ordering(D_map, ordering, T_p_map):
    n = len(ordering)
    T_p_all = sum(T_p_map[u] for u in ordering)
    T_h = sum(D_map[ordering[i]][ordering[(i + 1) % n]] for i in range(n))
    T_tour = 0.0
    if T_p_all < T_h:
        # single pass: init + collect at each vertex immediately
        for pos in range(n):
            node = ordering[pos]
            T_tour += T_p_map[node]
            T_tour += D_map[node][ordering[(pos + 1) % n]]
    else:
        # first trip: initialize tasks, travel only
        T_s = {}
        for pos in range(n):
            node = ordering[pos]
            T_s[node] = T_tour
            T_tour += D_map[node][ordering[(pos + 1) % n]]
        # second trip: collect outputs, waiting if needed
        for pos in range(n):
            node = ordering[pos]
            elapsed = T_tour - T_s[node]
            if T_p_map[node] > elapsed:
                T_tour += (T_p_map[node] - elapsed)
            T_tour += D_map[node][ordering[(pos + 1) % n]]
    return T_tour


# ============================================================
# 6. DP (Eq. 7)
# ============================================================

# ---------------- build H and P matrices for ordering ----------------
def build_H_P(ordering, D_map, T_p_map):
    n = len(ordering)
    travel_edge = [D_map[ordering[r]][ordering[(r + 1) % n]] for r in range(n)]
    # H[i][j] = sum_{r=i}^{j-1} travel_edge[r] for 0<=i<=j<=n
    H = [[0.0] * (n + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        s = 0.0
        for j in range(i + 1, n + 1):
            s += travel_edge[j - 1]
            H[i][j] = s
    # P[k][j] = sum T_p of ordering[k..j-1]
    P = [[0.0] * (n + 1) for _ in range(n)]
    for k in range(n):
        s = 0.0
        for j in range(k + 1, n + 1):
            s += T_p_map[ordering[j - 1]]
            P[k][j] = s
    return H, P

def dp_eq7(ordering, D_map, T_p_map, threshold):
    n = len(ordering)
    H, P = build_H_P(ordering, D_map, T_p_map)
    total_H = H[0][n]

    T1 = [[INF] * (n + 1) for _ in range(n + 1)]
    T2 = [[INF] * (n + 1) for _ in range(n + 1)]
    T = [[INF] * (n + 1) for _ in range(n + 1)]
    S = [[-1] * (n + 1) for _ in range(n + 1)]

    # initialise diagonals (i==i)
    for i in range(n + 1):
        T1[i][i] = 0.0
        T2[i][i] = 0.0
        T[i][i] = 0.0
        if i < n:
            S[i][i + 1] = i

    T_p_max = max(T_p_map.values()) if T_p_map else 0.0

    # DP loops (Eq.7 candidate)
    for l in range(1, n + 1):
        for i in range(0, n - l + 1):
            j = i + l
            S[i][j] = i
            T1[i][j] = INF
            T2[i][j] = INF
            T[i][j] = INF
            for k in range(i, j):
                # trip1 and trip2 per Eq.7 (with t(k,j)=H[k][j])
                trip1 = T1[i][k] + P[k][j] + H[k][j]
                trip2 = T2[i][k] + H[k][j]  # t(k,j) == H(k,j)
                # T_wk as in pseudocode: ensure consistent formula
                T_p_k = T_p_map[ordering[k]]
                term = total_H + P[k][j] - H[i][k] + T2[i][k]
                T_wk = max(T_p_k - term, 0.0)
                candidate = trip1 + trip2 + T_wk
                if candidate < T[i][j]:
                    T1[i][j] = trip1
                    T2[i][j] = trip2
                    T[i][j] = candidate
                    S[i][j] = k
            # fallback: enforce DP value <= provided threshold (heuristic_time)
            if T[i][j] > threshold:
                T1[i][j] = H[i][j]
                T2[i][j] = H[i][j]
                T[i][j] = threshold
                # set chain predecessors as in pseudocode
                m = j
                while m > i:
                    S[i][m] = m - 1
                    m -= 1

    info = {'T1': T1, 'T2': T2, 'H': H, 'P': P, 'threshold_used': threshold, 'total_H': total_H, 'T_p_max': T_p_max}
    return T, S#, info

def construct_tour(ordering, S, D_map, T_p_map):
    n = len(ordering)
    k = S[0][n]
    Tk = []
    while k != 0 and k != -1:
        Tk.insert(0, k)
        k = S[0][k]
    Tk.insert(0, 0)
    Tk_set = {ordering[t] for t in Tk}

    T_tour = 0
    start_times = {}

    # first trip
    for i in range(n):
        v = ordering[i]
        if v in Tk_set:
            start_times[v] = T_tour
        else:
            T_tour += T_p_map[v]
        T_tour += D_map[v][ordering[(i+1) % n]]

    # second trip
    for idx in Tk:
        v = ordering[idx]
        elapsed = T_tour - start_times[v]
        if T_p_map[v] > elapsed:
            T_tour += (T_p_map[v] - elapsed)

    return T_tour, Tk

# ============================================================
# 7. MAIN EXPERIMENT
# ============================================================
def run_experiment(tsplib_path):
    coords = load_tsplib_coords(tsplib_path)

    p = 0.75
    mu_node, sigma_node = 50000000, 3200000
    NUM = 10
    G = build_er_graph_from_coords(coords, p=p, seed=1000 )
    nodes, Dmat, D_map = compute_metric_closure_distances(G)
    ordering = compute_hamiltonian_cycle_from_metric(Dmat, nodes)

    results = []

    for g in range(NUM):
        #G = build_er_graph_from_coords(coords, p=p, seed=1000 + g)
        if G is None:
            continue

        # node service times (normally distributed, clipped >0)
        for v in G.nodes():
            G.nodes[v]["service"] = max(
                np.random.normal(mu_node, sigma_node), 0.01
            )

        #nodes, Dmat, D_map = compute_metric_closure_distances(G)
        #ordering = compute_hamiltonian_cycle_from_metric(Dmat, nodes)
        T_p_map = {v: G.nodes[v]["service"] for v in G.nodes()}

        heuristic = tsp_tet_heuristic_from_ordering(D_map, ordering, T_p_map)
        Tmat, Smat = dp_eq7(ordering, D_map, T_p_map, heuristic)
        dp_value = Tmat[0][len(ordering)]
        constructed, Tk = construct_tour(ordering, Smat, D_map, T_p_map)

        improvement = (heuristic - constructed) / heuristic * 100
        results.append(improvement)

        print(
            f"Run {g+1}: Heuristic={heuristic:.2f}, "
            f"DP={constructed:.2f}, Improvement={improvement:.2f}%"
        )

    print("\nSummary:")
    print(f"Average improvement: {np.mean(results):.2f}%")
    print(f"Max improvement:     {np.max(results):.2f}%")


# ============================================================
# RUN
# ============================================================
run_experiment("/content/dsj1000.tsp")