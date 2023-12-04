from numba import njit, int32
import numpy as np
from utils.assignment.exact import ob2dict


def md_primal_heuristics(data, ob, cost, truck_gar, num_cnt, obt, truck_time):
    remain = np.arange(len(ob))
    perm = np.random.permutation(len(truck_gar))
    obd, obc = ob2dict(ob)
    for truck in perm:
        garage = truck_gar[truck]
        truck_t = truck_time[truck]
        cost_cpy = np.zeros(len(cost))
        cost_cpy[num_cnt[garage]:num_cnt[garage + 1]] = cost[num_cnt[garage]:num_cnt[garage + 1]]
        cost_cpy[obt < truck_t] = 0
        idx = np.argmax(cost_cpy)
        route = trim_route(ob[idx])

    pass


@njit
def update_cardinality(data, cost, remain, ob, max_num):
    d = data.copy()
    remain_out = np.zeros(len(remain), dtype=int32)
    cnt = 0
    for idx in remain:
        for rid in range(max_num):
            node = ob[idx, rid]
            if not node:
                break
            d[node, 12] -= 1
            if d[node, 12] < 0:
                cost[idx] = 0
                break
        remain_out[cnt] = idx
        cnt += 1
    return remain_out[:cnt]


@njit
def update_selected(route, data, cost, idx):
    for i in range(len(route)):
        node = route[i]
        if not node:
            break
        data[node, 12] -= 1
        cost[idx] = 0


def trim_route(route):
    n = len(route)
    res = np.zeros(n)
    cnt = 0
    for i in range(n):
        if not route[i]:
            break
        res[i] = route[i]
        cnt += 1
    return res[:cnt]
