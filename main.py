import numpy as np

import pandas as pd
from utils.mode import get_py_files, mode_control
from utils.bundle.sifting import feasible_order_gen, feasible_order_pair_gen, row2col, cdfs_order_gen, \
    get_feasible_order_bundles, cdfs_order_gen_par, get_all_order_bundle_cost, cdfs_order_gen_py, \
    get_feasible_order_bundles_py, get_all_feasible_order_bundles, ob2dict

# files = get_py_files()
# mode_control(files, "debug")
data = np.loadtxt("data.npy")
dist_mat = np.loadtxt("dist_mat.npy")
time_mat = np.loadtxt("time_mat.npy")
postcode_arr = np.loadtxt("postcode_arr.npy").astype(int)

data_df = pd.DataFrame(data).rename(columns={
    0: "Distance",
    1: "TravellingTime",
    2: "EarlyTimeWindowOrigin",
    3: "LateTimeWindowOrigin",
    4: "EarlyTimeWindowDestination",
    5: "LateTimeWindowDestination",
    6: "LoadingTime",
    7: "UnloadingTime",
    8: "WaitingTimeOrigin",
    9: "WaitingTimeDestination",
    10: "Cost",
    11: "isImaginary"
})
data_df["pc1"] = postcode_arr[:, 0]
data_df["pc2"] = postcode_arr[:, 1]
data_unique = data_df.groupby(data_df.columns.tolist()).size().reset_index()
postcode_arr = data_unique[["pc1", "pc2"]].values.astype(int)
data_unique.drop(columns=["pc1", "pc2"], inplace=True)
data = data_unique.values
# for i in range(len(data)):
#     if data[i, 5] < data[i, 3]:
#         data[i, 5] = data[i, 4] + 8
etw_g = 6.
ltw_g = 8.
max_num = 6
max_node = 6
num_real_orders = len(data) - 5
route = np.zeros(1 + max_num, dtype=int)
route[0] = num_real_orders
num_nodes = 0
distance = np.zeros(2 * max_num + 1, dtype=int)
tw = np.zeros(2 * (max_num + 1), dtype=int)
garage = 0
udc = 0.43
utc = 0.48
shift = 12.
ltw_gr = ltw_g + shift

route, partial_duration_arr, total_duration_arr, partial_cost_saving_arr, total_cost_saving_arr = (
    feasible_order_gen(garage, num_real_orders, data, dist_mat, time_mat, postcode_arr, etw_g, ltw_g, utc, udc, shift))

route_pair, partial_duration_arr2, total_duration_arr2, partial_cost_saving_arr2, total_cost_saving_arr2 = (
    feasible_order_pair_gen(route, garage, data, dist_mat, time_mat, postcode_arr, ltw_g, utc, udc, shift,
                            partial_cost_saving_arr, partial_duration_arr, max_node, num_real_orders, etw_g, ltw_gr))

lookup_table, lookup_index = row2col(num_real_orders, route_pair)

res = np.zeros((100000, 6), dtype=int)
count = 0

rr = cdfs_order_gen(route_pair[0], lookup_table, lookup_index, 6, 6, data, time_mat, postcode_arr, garage, shift,
                    num_real_orders,
                    etw_g, ltw_gr)
# %timeit
# rr = cdfs_order_gen(route_pair[0], lookup_table, lookup_index, 6, 6, data, time_mat, postcode_arr, garage, shift,
#                     num_real_orders, etw_g, ltw_gr)
#
# %timeit
# rr = cdfs_order_gen_py(route_pair[0], lookup_table, lookup_index, 6, 6, data, time_mat, postcode_arr, garage, shift,
#                        num_real_orders, etw_g, ltw_gr)
#
# %timeit
# order_bundles_sg = get_feasible_order_bundles_py(route_pair, lookup_table, lookup_index, 6, max_num, data, time_mat,
#                                                  postcode_arr, garage, shift, num_real_orders, etw_g, ltw_gr)
#
# %timeit
# order_bundles_sg = get_feasible_order_bundles(route_pair, lookup_table, lookup_index, 6, max_num, data, time_mat,
#                                               postcode_arr, garage, shift, num_real_orders, etw_g, ltw_gr)
garage_arr = np.arange(5)
order_bundles, num_cnt = get_all_feasible_order_bundles(route_pair, lookup_table, lookup_index, max_node, max_num, data,
                                                        time_mat, postcode_arr, garage_arr, shift, num_real_orders,
                                                        etw_g, ltw_gr)
cost_saving = get_all_order_bundle_cost(order_bundles, garage_arr, num_cnt, data, dist_mat, postcode_arr, utc, udc)
obd, obc = ob2dict(order_bundles)

ob = order_bundles
cost = cost_saving
mode = "commercial"  # "free"#
truck_gar = {0: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 1, 6: 2, 7: 2, 8: 2, 9: 3, 10: 3, 11: 3, 12: 4, 13: 4, 14: 4, 15: 4}
obt = 8 * np.ones(len(ob))
truck_time = 6. + 0.002 * np.random.random(len(truck_gar))
# order_bundles_sg = get_feasible_order_bundles(route_pair, lookup_table, lookup_index, 6, max_num, data, time_mat, postcode_arr,
#                                            garage, shift, num_real_orders, etw_g, ltw_gr)
# cost_saving = get_all_order_bundle_cost(order_bundles_sg, garage, data, dist_mat, postcode_arr, utc, udc)
# print(cost_saving.mean())

# route_arr = route_extension(route_pair[0], lookup_table, res, 4)
# rrp = cdfs_order_gen_par(route_pair[0], lookup_table , lookup_index,6, 6, data, time_mat, postcode_arr, garage, shift,
# #                          num_real_orders, etw_g, ltw_gr)
# # # %timeit rrp = cdfs_order_gen_par(route_pair[0], lookup_table, lookup_index, 6, 6, data, time_mat, postcode_arr, garage, shift,num_real_orders, etw_g, ltw_gr)
