import numpy as np
from numba import njit, int32, prange, boolean


@njit
def feasible_order_gen(garage: int, num_real_orders: int, data: np.ndarray, dist_mat: np.ndarray, time_mat: np.ndarray,
                       postcode_arr: np.ndarray, etw_g: float, ltw_g: float, utc: float, udc: float, shift: float):
    res = np.zeros(num_real_orders, dtype=int32)
    count = 0
    partial_cost_saving_arr = np.zeros(num_real_orders)
    total_cost_saving_arr = np.zeros(num_real_orders)
    partial_duration_arr = np.zeros(num_real_orders)
    total_duration_arr = np.zeros(num_real_orders)
    for i in range(num_real_orders):
        deadhead_out_time = get_navi_info(time_mat, postcode_arr, garage, i)
        deadhead_out_dist = get_navi_info(dist_mat, postcode_arr, garage, i)
        deadhead_in_time = get_navi_info(time_mat, postcode_arr, i, garage)
        deadhead_in_dist = get_navi_info(time_mat, postcode_arr, i, garage)
        travel_time = data[i, 1]
        travel_dist = data[i, 0]
        load_time = data[i, 6]
        unload_time = data[i, 7]
        if etw_g + deadhead_out_time > data[i, 3]:
            continue
        if data[i, 4] + deadhead_in_time > ltw_g + shift:
            continue
        if travel_time + load_time + unload_time > shift:
            continue
        res[count] = i
        partial_duration_arr[count] = deadhead_out_time + travel_time + load_time + unload_time
        total_duration_arr[count] = partial_duration_arr[count] + deadhead_in_time
        partial_cost_saving_arr[count] = data[i, 10] - udc * deadhead_out_dist - utc * travel_dist
        total_cost_saving_arr[count] = partial_cost_saving_arr[count] - udc * deadhead_in_dist
        count += 1
    return res[:count], partial_duration_arr[:count], total_duration_arr[:count], partial_cost_saving_arr[
                                                                                  :count], total_cost_saving_arr[:count]


@njit(fastmath=True)
def feasible_order_pair_gen(fod: np.array, garage: int, data: np.ndarray, dist_mat: np.ndarray,
                            time_mat: np.ndarray, postcode_arr: np.ndarray, ltw_g: float, utc: float,
                            udc: float, shift: float, partial_cost_saving_arr: np.ndarray,
                            partial_duration_arr: np.ndarray, max_node: int, num_real_orders: int, etw_g: float,
                            ltw_gr: float):
    n = len(fod)
    max_num_pair = n ** 2
    res = np.zeros((max_num_pair, 2), dtype=int32)
    count = 0
    my_partial_cost_saving_arr = np.zeros(max_num_pair)
    my_total_cost_saving_arr = np.zeros(max_num_pair)
    my_partial_duration_arr = np.zeros(max_num_pair)
    my_total_duration_arr = np.zeros(max_num_pair)
    for i in fod:
        for j in fod:
            if i != j or data[i, 12] >= 2.:
                travel_time_j = data[j, 1]
                travel_dist_j = data[j, 0]
                load_time_j = data[j, 6]
                unload_time_j = data[j, 7]
                deadhead_in_time_j = get_navi_info(time_mat, postcode_arr, j, garage)
                deadhead_in_dist_j = get_navi_info(dist_mat, postcode_arr, j, garage)
                my_partial_duration = partial_duration_arr[i] + travel_time_j + load_time_j + unload_time_j
                my_total_duration = my_partial_duration + deadhead_in_time_j
                if my_total_duration > shift:
                    continue
                deadhead_time_ij = get_navi_info(time_mat, postcode_arr, i, j)
                deadhead_dist_ij = get_navi_info(dist_mat, postcode_arr, i, j)
                if data[i, 4] + deadhead_time_ij > data[j, 5]:
                    continue
                if data[j, 4] + deadhead_in_time_j > ltw_g + shift:
                    continue
                route = i * np.ones(1, dtype=int32)
                if not is_feasible(route, j, max_node, data, time_mat, postcode_arr, garage, shift, num_real_orders,
                                   etw_g, ltw_gr):
                    continue
                res[count, 0] = i
                res[count, 1] = j
                my_partial_duration_arr[count] = my_partial_duration
                my_total_duration_arr[count] = my_total_duration
                my_partial_cost_saving_arr[count] = partial_cost_saving_arr[
                                                        i] - utc * travel_dist_j - udc * deadhead_dist_ij
                my_total_cost_saving_arr[count] = my_partial_cost_saving_arr[count] - udc * deadhead_in_dist_j
                count += 1
    return (res[:count], my_partial_duration_arr[:count], my_total_duration_arr[:count],
            my_partial_cost_saving_arr[:count], my_total_cost_saving_arr[:count])


@njit
def get_navi_info(mat: np.ndarray, pca: np.ndarray, node1: int, node2: int):
    return mat[pca[node1, 1], pca[node2, 0]]


# data transformation
@njit
def row2col(num_real_orders: int, route_pair: np.ndarray):
    lookup_table = -np.ones((num_real_orders, num_real_orders), dtype=int32)
    lookup_index = np.zeros(num_real_orders, dtype=int32)
    prev = -1
    count = 0
    for i in range(len(route_pair)):
        start = route_pair[i, 0]
        end = route_pair[i, 1]
        if start != prev:
            lookup_index[prev] = count
            count = 0
            prev = start
        lookup_table[start, count] = end
        count += 1
    lookup_index[prev] = count
    return lookup_table, lookup_index


@njit
def is_feasible(route, node, max_node, data, time_mat, pca, garage, shift, num_real_orders, etw_g, ltw_gr):
    # # duplication is not allowed
    # for r in route:
    #     if r == node:
    #         return 0
    # duplication is allowed with restricted no. of duplicates
    route_node_cnt = get_route_node_cnt(route, num_real_orders)
    if not check_route_node(node, route_node_cnt, data):
        return 0
    # if get_total_travel_time(route, pca, data, time_mat, garage) > shift:
    #     return 0
    if len(route) > max_node - 1:
        return 0
    # if not check_time_window_propagation(route, pca, data, time_mat):
    #     return 0
    if not check_route_tw(etw_g, route, data, time_mat, pca, garage, ltw_gr, shift):
        return 0
    return 1


@njit(fastmath=True)
def get_total_travel_time(route, pca, data, time_mat, garage):
    deadhead_out_time = get_navi_info(time_mat, pca, garage, route[0])
    deadhead_in_time = get_navi_info(time_mat, pca, route[-1], garage)
    total_time = deadhead_out_time + deadhead_in_time
    for i in range(len(route) - 1):
        node = route[i]
        next_node = route[i + 1]
        total_time += data[node, 1] + data[node, 6] + data[node, 7]
        total_time += get_navi_info(time_mat, pca, node, next_node)
    return total_time


@njit
def check_time_window_propagation(route, pca, data, time_mat):
    for i in range(len(route) - 1):
        node = route[i]
        next_node = route[i + 1]
        if data[node, 4] + get_navi_info(time_mat, pca, node, next_node) + data[node, 6] > data[next_node, 5]:
            return 0
    return 1


@njit
def get_route_node_cnt(route, num_real_orders):
    res = np.zeros(num_real_orders)
    n = len(route)
    for i in range(n):
        res[route[i]] += 1
    return res


@njit
def check_route_node(node, route_node_cnt, data):
    if route_node_cnt[node] + 1 > data[node, 12]:
        return 0
    return 1


@njit(fastmath=True)
def check_route_tw(start, route, data, time_mat, pca, garage, ltw_gr, shift):
    # 2,3,4,5
    t, label = _check_route_tw(start, garage, route[0], data, time_mat, pca)
    if not label:
        return 0
    for i in range(1, len(route)):
        prev = route[i - 1]
        node = route[i]
        t, label = _check_route_tw(t, prev, node, data, time_mat, pca)
        if not label:
            return 0
    # check return time
    node = route[-1]
    if t + data[node, 7] + get_navi_info(time_mat, pca, node, garage) > ltw_gr:
        return 0
    if t - start > shift + 2:  # todo: constraint too tight
        return 0
    return 1


@njit(fastmath=True)
def _check_route_tw(eat, prev, node, data, time_mat, pca):
    t1 = max(eat + data[prev, 7] + get_navi_info(time_mat, pca, prev, node), data[node, 2])
    if t1 > data[node, 3]:
        return t1, 0
    t2 = max(t1 + data[node, 6] + data[node, 1], data[node, 4])
    if t2 > data[node, 5]:
        return t2, 0
    return t2, 1


#
# def route_extension(route, lookup_table, res, max_node):
#     # recursive route extension
#     global count
#     for node in lookup_table[route[-1]]:
#         if node == -1:
#             break
#         if is_feasible(route, node, max_node):
#             new_route = np.append(route, node)
#             res[count, :len(new_route)] = new_route
#             count += 1
#             route_extension(new_route, lookup_table, res, max_node)
#     return res
#
#
# def _route_extension(route, lookup_table, max_node):
#     # stack version of route extension
#     cnt = 0
#     res = np.zeros((1000000, 6), dtype=int)
#     stack = [route]
#     while stack:
#         route = stack.pop()
#         for node in lookup_table[route[-1]]:
#             if node == -1:
#                 break
#             if is_feasible(route, node, max_node):
#                 new_route = np.append(route, node)
#                 stack.append(new_route)
#                 res[cnt, :len(new_route)] = new_route
#                 cnt += 1
#     return res

@njit
def cdfs_order_gen(route, lookup_table, lookup_index, max_node, max_num, data, time_mat, pca, garage, shift,
                   num_real_orders, etw_g, ltw_gr):
    cnt = 0
    res = np.zeros((1000000, max_num), dtype=int32)  #
    stack = np.zeros((10000, max_num), dtype=int32)  #
    num = 1
    stack[0, :len(route)] = route
    rm = np.zeros(10000, dtype=int32)
    rm[0] = len(route)
    while num:
        num -= 1
        route = get_route(stack, num, rm)
        stack[num] = 0  # np.zeros(max_num, dtype=int32)
        end_node = route[-1]
        for j in range(lookup_index[end_node]):
            node = lookup_table[end_node, j]
            if is_feasible(route, node, max_node, data, time_mat, pca, garage, shift, num_real_orders, etw_g, ltw_gr):
                stack[num, :len(route)] = route
                stack[num, len(route)] = node
                rm[num] = len(route) + 1
                res[cnt, :] = stack[num, :].copy()
                num += 1
                cnt += 1
    return res[:cnt]


def cdfs_order_gen_py(route, lookup_table, lookup_index, max_node, max_num, data, time_mat, pca, garage, shift,
                      num_real_orders, etw_g, ltw_gr):
    cnt = 0
    res = np.zeros((1000000, max_num), dtype=int)  #
    stack = np.zeros((10000, max_num), dtype=int)  #
    num = 1
    stack[0, :len(route)] = route
    rm = np.zeros(10000, dtype=int)
    rm[0] = len(route)
    while num:
        num -= 1
        route = get_route(stack, num, rm)
        stack[num] = 0  # np.zeros(max_num, dtype=int32)
        end_node = route[-1]
        for j in range(lookup_index[end_node]):
            node = lookup_table[end_node, j]
            if is_feasible(route, node, max_node, data, time_mat, pca, garage, shift, num_real_orders, etw_g, ltw_gr):
                stack[num, :len(route)] = route
                stack[num, len(route)] = node
                rm[num] = len(route) + 1
                res[cnt, :] = stack[num, :].copy()
                num += 1
                cnt += 1
    return res[:cnt]


@njit
def check_fun(route, node_arr, max_node, data, time_mat, pca, garage, shift, num_real_orders, etw_g, ltw_gr):
    n = len(node_arr)
    res = np.zeros(n)
    for i in range(n):
        node = node_arr[i]
        b = is_feasible(route, node, max_node, data, time_mat, pca, garage, shift, num_real_orders, etw_g, ltw_gr)
        res[i] = b
    return res


@njit
def get_route(stack, num, rm):
    res = np.zeros(rm[num], dtype=int32)
    for i in range(rm[num]):
        res[i] = stack[num, i]
    return res


@njit(fastmath=True, parallel=1)
def cdfs_order_gen_par(route, lookup_table, lookup_index, max_node, max_num, data, time_mat, pca, garage, shift,
                       num_real_orders, etw_g, ltw_gr):
    cnt = 0
    res = np.zeros((1000000, max_num), dtype=int32)
    stack = np.zeros((10000, max_num), dtype=int32)
    num = 1
    stack[0, :len(route)] = route
    rm = np.zeros(10000, dtype=int32)
    rm[0] = len(route)
    while num:
        num -= 1
        route = stack[num, :rm[num]].copy()
        stack[num] = np.zeros(max_num, dtype=int32)
        num_iter = lookup_index[route[-1]]
        tmp = np.zeros((num_iter, max_num), dtype=int32)  # pre-allocation: stack
        indicator = np.zeros(num_iter, dtype=boolean)  # pre-allocation: feasibility check
        prev_node = route[-1]
        for j in prange(num_iter):
            node = lookup_table[prev_node, j]
            indicator[j] = is_feasible(route, node, max_node, data, time_mat, pca, garage, shift, num_real_orders,
                                       etw_g, ltw_gr)
            new_route = np.append(route, node)
            tmp[j, :len(new_route)] = new_route
        num_feasible = sum(indicator)
        stack[num:num + num_feasible] = tmp[indicator]
        rm[num:num + num_feasible] = len(route) + 1
        res[cnt:cnt + num_feasible] = tmp[indicator]
        num += num_feasible
        cnt += num_feasible
    return res[:cnt]


@njit(fastmath=True)
def get_feasible_order_bundles(route_pair, lookup_table, lookup_index, max_node, max_num, data, time_mat, pca, garage,
                               shift, num_real_orders, etw_g, ltw_gr):
    order_bundles = np.zeros((10000000, max_num), dtype=int32)
    ptr = 0
    for i in range(len(route_pair)):
        rr = cdfs_order_gen(route_pair[i], lookup_table, lookup_index, max_node, max_num, data, time_mat, pca, garage,
                            shift, num_real_orders, etw_g, ltw_gr)
        n = len(rr)
        order_bundles[ptr:ptr + n, :] = rr
        ptr += n
    return order_bundles[:ptr]


def get_feasible_order_bundles_py(route_pair, lookup_table, lookup_index, max_node, max_num, data, time_mat, pca,
                                  garage, shift, num_real_orders, etw_g, ltw_gr):
    order_bundles = np.zeros((10000000, max_num), dtype=int)
    ptr = 0
    for i in range(len(route_pair)):
        rr = cdfs_order_gen_py(route_pair[i], lookup_table, lookup_index, max_node, max_num, data, time_mat, pca,
                               garage,
                               shift, num_real_orders, etw_g, ltw_gr)
        n = len(rr)
        order_bundles[ptr:ptr + n] = rr
        ptr += n
    return order_bundles[:ptr]


def get_all_feasible_order_bundles(route_pair, lookup_table, lookup_index, max_node, max_num, data, time_mat, pca,
                                   garage_arr, shift, num_real_orders, etw_g, ltw_gr):
    num_gar = len(garage_arr)
    res = np.zeros((10000000, max_num), dtype=int)
    cnt = 0
    num_cnt = np.zeros(num_gar + 1, dtype=int)
    for i in range(num_gar):
        garage = garage_arr[i]
        ob = get_feasible_order_bundles_py(route_pair, lookup_table, lookup_index, max_node, max_num, data, time_mat,
                                           pca, garage, shift, num_real_orders, etw_g, ltw_gr)
        n = len(ob)
        res[cnt:cnt + n] = ob
        cnt += n
        num_cnt[i + 1] = cnt
    return res[:cnt], num_cnt


@njit(fastmath=True)
def get_order_bundle_cost(route, garage, data, dist_mat, pca, utc, udc):
    cost_saving = 0
    deadhead = 0
    travel = 0
    node = route[0]
    cost_saving += data[node, 10]
    deadhead += get_navi_info(dist_mat, pca, garage, node)
    travel += data[node, 0]
    for i in range(len(route) - 1):
        if not route[i + 1]:
            break
        deadhead += get_navi_info(dist_mat, pca, route[i], route[i + 1])
        travel += data[route[i + 1], 0]
        cost_saving += data[route[i + 1], 10]
    deadhead += get_navi_info(dist_mat, pca, route[-1], garage)
    cost_saving -= travel * utc + deadhead * udc
    return cost_saving


@njit
def get_order_bundle_cost_sg(order_bundle, garage, data, dist_mat, pca, utc, udc):
    num_ob = len(order_bundle)
    cost_saving = np.zeros(num_ob)
    for i in range(num_ob):
        cost_saving[i] = get_order_bundle_cost(order_bundle[i], garage, data, dist_mat, pca, utc, udc)
    return cost_saving


@njit
def get_all_order_bundle_cost(order_bundle, garage_arr, num_cnt, data, dist_mat, pca, utc, udc):
    n = len(garage_arr)
    res = np.zeros(len(order_bundle))
    for i in range(n):
        order_bundles = order_bundle[num_cnt[i]:num_cnt[i + 1]]
        garage = garage_arr[i]
        cost = get_order_bundle_cost_sg(order_bundles, garage, data, dist_mat, pca, utc, udc)
        res[num_cnt[i]:num_cnt[i + 1]] = cost
    return res


def ob2dict(ob):
    res = dict()
    cnt = dict()
    n = len(ob)
    for i in range(n):
        for node in ob[i]:
            if not node:
                break
            if node not in res:
                res[node] = [i]
                cnt[node] = {i: 1}
            else:
                res[node].append(i)
                if i in cnt[node]:
                    cnt[node][i] += 1
                else:
                    cnt[node][i] = 1
    return res, cnt