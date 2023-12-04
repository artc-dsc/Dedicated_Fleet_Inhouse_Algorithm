import numpy as np
from mip import Model, quicksum, maximize, INTEGER, GRB, CBC, LinExpr, BINARY
from utils.bundle.sifting import ob2dict


# def exact_md_ob_assignment(data, ob, cost, mode="free"):
#     solver_name = GRB if mode == "commercial" else CBC
#     m = Model(name="assignment", solver_name=solver_name)
#     n = len(ob)
#     x = [m.add_var(name=f"x_{i}", var_type=INTEGER) for i in range(n)]
#     obj = quicksum(x[i] * cost[i] for i in range(n))
#     m.objective = maximize(obj)
#     # num of duplication constraints
#     obd, obc = ob2dict(ob)
#     cardinality_lhs = dict()
#     for node, rid in obd.items():
#         cardinality_lhs[node] = quicksum(x[i] * obc[node][i] for i in rid)
#     for node, expr in cardinality_lhs.items():
#         m.add_constr(expr <= data[node, 12])
#     m.optimize()


def exact_md_ob_truck_assignment(data, ob, cost, truck_gar, num_cnt, obt, truck_time, mode="free"):
    solver_name = GRB if mode == "commercial" else CBC
    m = Model(name="assignment", solver_name=solver_name)
    n = len(ob)
    x, xr = get_decision_variable(m, truck_gar, num_cnt, obt, truck_time)
    obj = quicksum(x[i, j] * cost[j] for i, j in x)
    m.objective = maximize(obj)
    # num of duplication constraints
    obd, obc = ob2dict(ob)
    cardinality_lhs = dict()
    for node, rid in obd.items():
        idx = [(i, j) for i in truck_gar for j in set(rid).intersection(set(xr[i]))]
        cardinality_lhs[node] = quicksum(x[i, j] * obc[node][j] for i, j in idx)
        # cardinality_lhs[node] = quicksum(x[i, j] * obc[node][j] for i, j in x if j in rid)
    for node, expr in cardinality_lhs.items():
        m.add_constr(expr <= data[node, 12])
    for truck in xr:
        m.add_constr(quicksum(x[truck, j] for j in xr[truck]) <= 1)
    m.optimize()
    res = []
    for i, j in x:
        if x[i, j].x >= 0.99:
            res.append((i, j))
            print(ob[j], cost[j])


def get_cardinality_idx(obd, truck_gar, xr):
    res = []
    for node, rid in obd.items():
        for i in truck_gar:
            j_list = set(xr[i]).intersection(set(rid))
            for j in j_list:
                res.append((i, j))


def get_decision_variable(m: Model, truck_gar: dict, num_cnt: np.ndarray, obt: np.ndarray, truck_time: dict):
    x = dict()
    xr = dict()
    for truck, garage in truck_gar.items():
        truck_t = truck_time[truck]
        for oid in range(num_cnt[garage], num_cnt[garage + 1]):
            if truck_t <= obt[oid]:
                x[truck, oid] = m.add_var(name=f"x_{truck}_{oid}", var_type=BINARY)
                if truck in xr:
                    xr[truck].append(oid)
                else:
                    xr[truck] = [oid]
    return x, xr

# def ob2dict(ob):
#     res = dict()
#     n = len(ob)
#     for i in range(n):
#         tmp = dict()
#         for node in ob[i]:
#             if not node:
#                 break
#             if node not in tmp:
#                 tmp[node] = 1
#             else:
#                 tmp[node] += 1
#         res[i] = tmp
#     return res
