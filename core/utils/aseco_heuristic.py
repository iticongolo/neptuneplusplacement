import copy
import json
import time

import numpy as np
from core import setup_runtime_data, setup_community_data, Data
node_cpu_usage = None
node_gpu_usage = None
node_memory_usage = None
# from simulations.json_file_hotel import JsonfileHotel as app
# from simulations.json_file_sockshop import JsonfileSockshop as app
# from simulations.json_file_complex import JsonfileComplex as app
from simulations.json_file_test import JsonfileTest as app
from core import Parameters as param

input = app.input


class ASECO:

    def __init__(self, data):
        self.cfj = np.zeros((len(data.functions), len(data.nodes)))
        self.node_cpu_available = copy.deepcopy(data.node_cores)
        self.node_memory_available = copy.deepcopy(data.node_memories)
        self.data = data
        self.w = []
        self.x = []
        self.y = []
        self.z = []
        self.coldstart = 0
        self.total_delay = 0
        self.network_delay = 0
    # def f_ordered_by_worklod(self, workload):
    #     ordered_f = np.zeros(len(workload))
    #     sorted_indices = np.argsort(workload[:, 1])
    #     A_sorted = A[sorted_indices]
    #     first_column_values = A_sorted[:, 0]
    #     print("Ordered array by the second column:")
    #     print(first_column_values)


    # note SELECTED
    def get_functions(self):
        function = []
        for func in self.data.functions:
            function.append(func.split("/")[1])
        return function

# note SELECTED
    # def get_w(data, x, y):
    #     fill_value = "0.0"
    #     # Initialize the array
    #     w = np.full((len(data.functions), len(data.nodes)), fill_value)
    #     w = np.empty((len(data.functions), len(data.nodes)), dtype=object)
    #     # w = np.zeros((len(data.functions), len(data.nodes)))
    #     function = get_functions(data)
    #     for f in range(len(data.functions)):
    #         for j in range(len(data.nodes)):
    #             express1 = "0.0"
    #             express2 = "0.0"
    #             for i in range(len(data.nodes)):
    #                 express1 = express1  + '+' + str(x[i, f, j]) + '*' + str(data.workload_on_source_matrix[f, i])
    #             for g in data.dag.get_predecessors_indexes(function[f]):
    #                 for k in range(len(data.nodes)):
    #                     express2 = express2 + '+' + str(data.m[g][f]) + '*' + w[g, k] + '*' + str(y[g, k, f, j])
    #             w[f, j] = express1 + '+' + express2
    #             # w[f, j] = express1 + express2
    #             print(w[f, j])
    #     return w

    # def get_w(data, x, y):
    #     w = np.zeros((len(data.functions), len(data.nodes)))
    #     function = get_functions(data)
    #     for f in range(len(data.functions)):
    #         for j in range(len(data.nodes)):
    #             express1 = 0.0
    #             express2 = 0.0
    #             for i in range(len(data.nodes)):
    #                 express1 = express1 + x[f, i,  j] * data.workload_on_source_matrix[f, i]
    #             for g in data.dag.get_predecessors_indexes(function[f]):
    #                 for k in range(len(data.nodes)):
    #                     express2 = express2 + data.m[g][f] * w[g, k] * y[g, k, f, j]
    #             w[f, j] = express1 + express2
    #     return w


    # def get_w_op(data, x):
    #     function = get_functions(data)
    #
    #     qty_f = len(data.functions)
    #     qty_nodes = len(data.nodes)
    #
    #     w = np.zeros((qty_f, qty_nodes))
    #
    #     shape_x = (qty_f, qty_nodes, qty_nodes)
    #     shape_y = (qty_f, qty_nodes, qty_f, qty_nodes)
    #
    #     x_array_size = np.ravel_multi_index((qty_f-1, qty_nodes-1,  qty_nodes-1), shape_x)+1
    #
    #     for f in range(qty_f):
    #         for j in range(qty_nodes):
    #             express1 = 0.0
    #             express2 = 0.0
    #             for i in range(qty_nodes):
    #                 index_x = np.ravel_multi_index((f, i, j), shape_x)
    #                 express1 = express1 + x[index_x] * data.workload_on_source_matrix[f, i]
    #             for g in data.dag.get_predecessors_indexes(function[f]):
    #                 for k in range(qty_nodes):
    #                     index_y = x_array_size+np.ravel_multi_index((g, k, f, j), shape_y)
    #                     express2 = express2 + data.m[g][f] * w[g, k] * x[index_y]
    #             w[f, j] = express1 + express2
    #     return w

    #note KEEP
    # def get_cpu_usage(data, x, node):
    #     lamb = data.workload_on_source_matrix
    #     u = data.core_per_req_matrix
    #     U = data.node_cores
    #     m=data.m
    #     w = get_w(data, x)
    #     x_array_size, y_array_size, z_array_size, shape_x, shape_y_z = get_size_vars(data)
    #     qty_f = len(data.functions)
    #     qty_nodes = len(data.nodes)
    #     shape_x = (qty_f, qty_nodes,  qty_nodes)
    #     dag = data.dag
    #     function = get_functions(data)
    #
    #     cpu_node_usage = 0
    #     for f in range(qty_f):
    #         for i in range(qty_nodes):
    #             index = np.ravel_multi_index((f, i, node), shape_x)
    #             cpu_node_usage += x[index]*lamb[f, i]*u[f, node]
    #             seq_successor = dag.get_sequential_successors_indexes(function[f])
    #             for fs in seq_successor:
    #                 index = x_array_size + np.ravel_multi_index((f, i, fs, node), shape_y_z)
    #                 cpu_node_usage += x[index]*m[f][fs]*u[fs, node]*w[f, i]
    #             parallel_successors_groups = dag.get_parallel_successors_indexes(function[f])
    #             for par_group in parallel_successors_groups:
    #                 for fp in par_group:
    #                     index = x_array_size + y_array_size + np.ravel_multi_index((f, i, fp, node), shape_y_z)
    #                     cpu_node_usage += x[index] * m[f][fp] * u[fp, node] * w[f, i]
    #     cpu_node_usage += cpu_node_usage
    #     return cpu_node_usage


    # note SELECTED

    # def object_function_heuristic(self,w,x, y, z, lambd = np.array([]), instances=np.array([])):
    #     function = self.get_functions()
    #     qty_f = len(self.data.functions)
    #     qty_nodes = len(self.data.nodes)
    #     dag = self.data.dag
    #     m = self.data.m
    #     nrt = self.data.nrt
    #     network_delay = self.data.node_delay_matrix
    #     lamb = self.data.workload_on_source_matrix
    #
    #     if len(lambd) > 0:
    #         lamb = lambd
    #
    #     # print(f'lamb ={lamb }')
    #     cfj_temp = copy.deepcopy(self.cfj)
    #     if len(instances) > 0:
    #         cfj_temp = copy.deepcopy(instances)
    #     function_cold_starts = self.data.function_cold_starts
    #     # print(f'function_cold_starts={function_cold_starts}')
    #
    #     for f in range(qty_f):
    #         temp_network_delay = 0
    #         temp_coldstart = 0
    #         temp_total_delay = 0
    #         for i in range(qty_nodes):
    #             if lamb[f, i] > 0:
    #                 for j in range(qty_nodes):
    #                     temp_network_delay = temp_network_delay + network_delay[i][j] * x[f, i, j] * lamb[f, i]
    #
    #             if cfj_temp[f, i]:
    #                 temp_coldstart = temp_coldstart + function_cold_starts[f]
    #                 cfj_temp[f, i] = 0
    #                 temp_total_delay = temp_total_delay + temp_network_delay + temp_coldstart
    #
    #             for j in range(qty_nodes):
    #                 sum_fs = 0
    #                 seq_successor = dag.get_sequential_successors_indexes(function[f])
    #                 for fs in seq_successor:
    #                     # delay_y = y[f, i, fs, j]*m[f][fs]*w[f, i]*nrt[fs]
    #                     delay_y = y[f, i, fs, j] * m[f][fs] * w[f, i]
    #                     if cfj_temp[fs, j]:
    #                         # print(f'function_cold_starts={function_cold_starts}')
    #                         temp_coldstart = temp_coldstart + function_cold_starts[fs]
    #                         temp_total_delay = temp_total_delay + function_cold_starts[fs]
    #                         cfj_temp[fs, j] = 0
    #                     sum_fs = sum_fs + delay_y
    #                 if sum_fs * network_delay[i][j] > 0:
    #                     temp_network_delay = temp_network_delay + sum_fs * network_delay[i][j]
    #                     temp_total_delay = temp_total_delay + sum_fs * network_delay[i][j]
    #             parallel_successors_groups = dag.get_parallel_successors_indexes(function[f])
    #
    #             for par_group in parallel_successors_groups:
    #                 max_network_delay_z = float('-inf')
    #                 max_total_delay_z = float('-inf')
    #                 max_coldstart = float('-inf')
    #                 for fp in par_group:
    #                     for j in range(qty_nodes):
    #                         # delay_z = z[f, i, fp, j] * m[f][fp] * w[f, i] * nrt[fp] * network_delay[i][j]
    #                         delay_z = z[f, i, fp, j] * m[f][fp] * w[f, i] * network_delay[i][j]
    #                         if delay_z > max_network_delay_z:
    #                             max_network_delay_z = delay_z
    #                         if cfj_temp[fp, j]:
    #                             if max_coldstart < function_cold_starts[fp]:
    #                                 max_coldstart = function_cold_starts[fp]
    #                             network_cold_delay = function_cold_starts[fp] + delay_z
    #                             if max_total_delay_z < network_cold_delay:
    #                                 max_total_delay_z = network_cold_delay
    #
    #                 if max_network_delay_z > float('-inf'):
    #                     temp_network_delay = temp_network_delay + max_network_delay_z
    #                 if max_coldstart > float('-inf'):
    #                     temp_coldstart = temp_coldstart + max_coldstart
    #                 if max_total_delay_z > float('-inf'):
    #                     temp_total_delay = temp_total_delay + max_total_delay_z
    #
    #                 for fp in par_group:
    #                     for j in range(qty_nodes):
    #                         cfj_temp[fp, j] = 0
    #         self.coldstart = self.coldstart + temp_coldstart
    #         self.network_delay = self.network_delay + temp_network_delay
    #         self.total_delay = self.total_delay + temp_total_delay
    #     return self.total_delay, self.coldstart, self.network_delay


    # note SELECTED

    # def object_function_heuristic (self,w,x, y, z, lambd = np.array([]), instances=np.array([])):
    #     function = self.get_functions()
    #     qty_f = len(self.data.functions)
    #     qty_nodes = len(self.data.nodes)
    #     dag = self.data.dag
    #     m = self.data.m
    #     network_delay = self.data.node_delay_matrix
    #     lamb = self.data.workload_on_source_matrix
    #
    #     if len(lambd) > 0:
    #         lamb = lambd
    #
    #     # print(f'lamb ={lamb }')
    #     cfj_temp = copy.deepcopy(self.cfj)
    #     if len(instances) > 0:
    #         cfj_temp = copy.deepcopy(instances)
    #
    #     for f in range(qty_f):
    #         temp_network_delay = 0
    #         temp_total_delay = 0
    #         for i in range(qty_nodes):
    #             if lamb[f, i] > 0:
    #                 for j in range(qty_nodes):
    #                     temp_network_delay = temp_network_delay + network_delay[i][j] * x[f, i, j] * lamb[f, i]
    #
    #             if cfj_temp[f, i]:
    #                 cfj_temp[f, i] = 0
    #                 temp_total_delay = temp_total_delay + temp_network_delay
    #
    #             for j in range(qty_nodes):
    #                 sum_fs = 0
    #                 seq_successor = dag.get_sequential_successors_indexes(function[f])
    #                 for fs in seq_successor:
    #                     # delay_y = y[f, i, fs, j]*m[f][fs]*w[f, i]*nrt[fs]
    #                     delay_y = y[f, i, fs, j] * m[f][fs] * w[f, i]
    #                     if cfj_temp[fs, j]:
    #                         cfj_temp[fs, j] = 0
    #                     sum_fs = sum_fs + delay_y
    #                 if sum_fs * network_delay[i][j] > 0:
    #                     temp_network_delay = temp_network_delay + sum_fs * network_delay[i][j]
    #                     temp_total_delay = temp_total_delay + sum_fs * network_delay[i][j]
    #             parallel_successors_groups = dag.get_parallel_successors_indexes(function[f])
    #
    #             for par_group in parallel_successors_groups:
    #                 max_network_delay_z = float('-inf')
    #                 max_total_delay_z = float('-inf')
    #                 max_coldstart = float('-inf')
    #                 for fp in par_group:
    #                     for j in range(qty_nodes):
    #                         delay_z = z[f, i, fp, j] * m[f][fp] * w[f, i] * network_delay[i][j]
    #                         if delay_z > max_network_delay_z:
    #                             max_network_delay_z = delay_z
    #
    #                 if max_network_delay_z > float('-inf'):
    #                     temp_network_delay = temp_network_delay + max_network_delay_z
    #                 if max_total_delay_z > float('-inf'):
    #                     temp_total_delay = temp_total_delay + max_total_delay_z
    #
    #                 for fp in par_group:
    #                     for j in range(qty_nodes):
    #                         cfj_temp[fp, j] = 0
    #
    #         self.network_delay = self.network_delay + temp_network_delay
    #         self.total_delay = self.coldstart + self.network_delay #self.total_delay + temp_total_delay
    #     return self.total_delay, self.coldstart, self.network_delay

    def object_function_heuristic(self, w, x, y, z, lambd=np.array([]), instances=np.array([])):
        function = self.get_functions()
        qty_f = len(self.data.functions)
        qty_nodes = len(self.data.nodes)
        dag = self.data.dag
        m = self.data.m
        network_delay = self.data.node_delay_matrix
        lamb = self.data.workload_on_source_matrix

        if len(lambd) > 0:
            lamb = lambd

        # print(f'lamb ={lamb }')
        cfj_temp = copy.deepcopy(self.cfj)
        if len(instances) > 0:
            cfj_temp = copy.deepcopy(instances)
        networkd=0
        for f in range(qty_f):
            temp_network_delay = 0
            temp_total_delay = 0
            for i in range(qty_nodes):
                if lamb[f, i] > 0:
                    for j in range(qty_nodes):
                        temp_network_delay = temp_network_delay + network_delay[i][j] * x[f, i, j] * lamb[f, i]


                if cfj_temp[f, i]:
                    cfj_temp[f, i] = 0
                    temp_total_delay = temp_total_delay + temp_network_delay

                for j in range(qty_nodes):
                    sum_fs = 0
                    seq_successor = dag.get_sequential_successors_indexes(function[f])
                    for fs in seq_successor:
                        # delay_y = y[f, i, fs, j]*m[f][fs]*w[f, i]*nrt[fs]
                        delay_y = y[f, i, fs, j] * m[f][fs] * w[f, i]
                        if cfj_temp[fs, j]:
                            cfj_temp[fs, j] = 0
                        sum_fs = sum_fs + delay_y
                    if sum_fs * network_delay[i][j] > 0:
                        temp_network_delay = temp_network_delay + sum_fs * network_delay[i][j]

                parallel_successors_groups = dag.get_parallel_successors_indexes(function[f])

                for par_group in parallel_successors_groups:
                    max_network_delay_z = float('-inf')
                    for fp in par_group:
                        delay_z = 0
                        for j in range(qty_nodes):
                            delay_z = delay_z + z[f, i, fp, j] * m[f][fp] * w[f, i] * network_delay[i][j]
                        if delay_z > max_network_delay_z:
                            max_network_delay_z = delay_z

                    if max_network_delay_z > float('-inf'):
                        temp_network_delay = temp_network_delay + max_network_delay_z
                    print(f'max_network_delay_z[{i}] = {max_network_delay_z}')
                    for fp in par_group:
                        for j in range(qty_nodes):
                            cfj_temp[fp, j] = 0
            print(f'ABB={networkd} + {temp_network_delay}')
            self.network_delay = self.network_delay+ temp_network_delay

            self.total_delay = self.coldstart + self.network_delay  # self.total_delay + temp_total_delay

        return self.total_delay, self.coldstart, self.network_delay

    def get_closest_available_node(self, f, sf, mf, i, perc_used_cpu, perc_cpu_diff_usage):  # TODO
        selected_node = -1
        cpu = 0.0
        memory = 0.0
        min_delay = float('inf')
        nodes = self.data.nodes
        candidate_nodes = []
        less_used_nodes = []

        max_delay_f = self.data.max_delay_matrix
        node_delay = self.data.node_delay_matrix
        f_placed = False
        # prepare a list of nodes with available cores and memory
        for j in range(len(nodes)):
            # print(f'if node_cpu_available[{j}] > 0 and node_memory_available[{j}] >= mf[{f}]:')
            # print(f'if {self.node_cpu_available[j]}> 0 and {self.node_memory_available[j]} >= {mf[sf]}')
            if self.node_cpu_available[j] > 0 and self.node_memory_available[j] >= mf[sf]:
                candidate_nodes.append(j)

        previous_nodes = self.get_nodes(sf)
        # print(f'previous_nodes[{f}]={previous_nodes}')
        if len(previous_nodes) > 0:
            for p_node in previous_nodes:
                if self.node_cpu_available[p_node] > 0:
                    candidate_nodes.append(p_node)
                    f_placed = True

        # prepare a list of nodes which the usage difference compared to the present node_i
        # is greater or equal to the acceptable percentage
        # for j in candidate_nodes:
        #     if perc_used_cpu[i]-perc_used_cpu[j] >= perc_cpu_diff_usage:
        #         print(f'perc_used_cpu[{i}]-perc_used_cpu[{j}]/{perc_used_cpu[i]}-{perc_used_cpu[j]}]={perc_used_cpu[i]-perc_used_cpu[j]}')
        #         less_used_nodes.append(j)

        # if no greater differences we select the present node_i if available or anyone with min network delay
        if len(less_used_nodes) == 0:
            if i in candidate_nodes:
                selected_node = i
            else:
                for j in candidate_nodes:
                    if min_delay > node_delay[i][j]:
                        min_delay = node_delay[i][j]
                        selected_node = j
                if selected_node >= 0:

                    # select all closest nodes
                    candidates = []
                    for j in candidate_nodes:
                        if node_delay[i][selected_node] == node_delay[i][j]:
                            candidates.append(j)
                    powerful_node_capacity = 0
                    used_node = -1
                    node_found = False

                    # select the node where the instance is already placed to only increase the workload and save energy
                    for node in candidates:
                        if self.cfj[sf, node]:
                            selected_node = node
                            node_found = True
                            break
                        if self.as_instance(node):
                            used_node = node
                    if not node_found:
                        if used_node > -1:
                            selected_node = used_node

                        # select a more powerfull nodes to avoid more nodes usage
                        else:
                            for node in candidates:
                                if self.node_cpu_available[node] > powerful_node_capacity:
                                    powerful_node_capacity = self.node_cpu_available[node]
                                    selected_node = node

        # If the list of less used nodes is not empty then we select the least used node
        else:
            the_least_used = -1
            diff_usage = 0
            for j in less_used_nodes:
                for k in less_used_nodes:
                    diff = perc_used_cpu[j] - perc_used_cpu[k]
                    if diff > diff_usage:
                        diff_usage = diff
                        the_least_used = k

            if diff_usage >= perc_cpu_diff_usage:
                selected_node = the_least_used
            else:
                for j in less_used_nodes:
                    if min_delay > node_delay[i][j]:
                        min_delay = node_delay[i][j]
                        selected_node = j
            if max_delay_f[f] < node_delay[i][selected_node]:
                if i in candidate_nodes:
                    selected_node = i

        if selected_node >= 0:
            cpu = self.node_cpu_available[selected_node]
            memory = self.node_memory_available[selected_node]
        # print(f'f, i={f}, {i}')
        # print(f'candidate_nodes={candidate_nodes}')
        #
        # print(f'selected_node={selected_node}')
        # print(f'node_cpu_available={self.node_cpu_available}')
        # print(f'node_memory_available={self.node_memory_available}')
        return selected_node, cpu, memory, f_placed


    # note SELECTED
    def fill_x(self, cpu_diff_usage=0.2):

        x = np.zeros((len(self.data.functions), len(self.data.nodes), len(self.data.nodes)))
        w = np.zeros((len(self.data.functions), len(self.data.nodes)))
        workload = self.data.workload_on_source_matrix
        perc_used_cpu = np.zeros(len(self.data.nodes))
        # self.cfj = np.zeros((len(self.data.functions), len(self.data.nodes)), dtype=bool)
        _, functions, _, mf, nodes, ufj, node_cpu = self.basic_fill_data()
        # ordered_f = f_ordered_by_worklod(workload)
        for f in range(len(functions)):
            for i in range(len(nodes)):
                lamb = workload[f][i]
                if lamb > 0:
                    cpu_requested = lamb*ufj[f, i]  # here xfij is 1
                    cpu_requested1 = copy.deepcopy(cpu_requested)
                    memory_requested = mf[f]  # here xfij is 1
                    memory_requested1 = copy.deepcopy(memory_requested)
                    allocation_finished = False
                    while cpu_requested1 > 0:  # we can use cpu_requested or memory_requested
                        if allocation_finished:
                            break

                        closest_node, _, _, f_placed = self.get_closest_available_node(f, f, mf, i, perc_used_cpu, cpu_diff_usage)

                        if closest_node < 0:
                            raise Exception("The nodes are overloaded, no more resources to be allocated!")
                        diff_cpu = self.node_cpu_available[closest_node] - cpu_requested1
                        diff_memory = self.node_memory_available[closest_node]
                        if not f_placed:
                            diff_memory = self.node_memory_available[closest_node] - memory_requested1  # get_closest_available_node
                        # returns node with enough memory
                        if diff_cpu >= 0:
                            if cpu_requested1 == cpu_requested:
                                x[f, i, closest_node] = 1.0
                            else:
                                x[f, i, closest_node] = cpu_requested1/cpu_requested  # x[f, i, closest_node] +

                            self.node_cpu_available[closest_node] = diff_cpu
                            w[f, closest_node] = w[f, closest_node] + x[f, i, closest_node] * lamb
                            perc_used_cpu[closest_node] = perc_used_cpu[closest_node] + cpu_requested1 / node_cpu[
                                closest_node]
                            allocation_finished = True
                        else:
                            perc_cpu = self.node_cpu_available[closest_node] / cpu_requested
                            self.node_cpu_available[closest_node] = 0.0
                            x[f,i,closest_node] = perc_cpu
                            cpu_requested1 = cpu_requested * (1 - perc_cpu)
                            w[f, closest_node] = w[f, closest_node] + x[f, i, closest_node] * lamb
                            perc_used_cpu[closest_node] = perc_used_cpu[closest_node] + cpu_requested1 / node_cpu[
                                closest_node]
                        if not self.cfj[f, closest_node]:
                            self.node_memory_available[closest_node] = diff_memory
                            self.cfj[f, closest_node] = True
        # print(f'W-x={w}')
        return x, w,  perc_used_cpu

    def fill_y(self, f, y, w, perc_used_cpu, perc_cpu_diff_usage):
        dag, functions, m, mf, nodes, ufj, node_cpu = self.basic_fill_data()
        seq_successor = dag.get_sequential_successors_indexes(functions[f])
        if len(seq_successor) > 0:
            for i in range(len(nodes)):
                omega = w[f, i]
                if omega > 0:
                    for fs in seq_successor:
                        omega1 = copy.deepcopy(omega)
                        for j in range(len(nodes)):
                            if omega1 > 0:
                                cpu_requested = omega1 * m[f][fs] * ufj[fs, j]  # here y[f,i,fs,j] is 1
                                memory_requested = mf[fs]  ## here y[f,i,fs,j] is 1
                                cpu_requested1 = copy.deepcopy(cpu_requested)
                                allocation_finished = False
                                while cpu_requested1 > 0:  # we can use cpu_requested or memory_requested
                                    if allocation_finished:
                                        break
                                    closest_node, _, _, f_placed = self.get_closest_available_node(f, fs, mf, i, perc_used_cpu, perc_cpu_diff_usage)
                                    if closest_node < 0:
                                        raise Exception("The nodes are overloaded, no more resources to be allocated!")
                                    diff_cpu = self.node_cpu_available[closest_node] - cpu_requested1
                                    # get_closest_available_node returns node with enough memory
                                    diff_memory = self.node_memory_available[closest_node]
                                    if not f_placed:
                                        diff_memory = self.node_memory_available[closest_node] - memory_requested  # get_closest_available_node
                                    if diff_cpu >= 0:
                                        if cpu_requested1 == cpu_requested:
                                            y[f, i, fs, closest_node] = 1.0
                                        else:
                                            y[f, i, fs, closest_node] = cpu_requested1 / cpu_requested # y[f, i, fs, closest_node] +

                                        self.node_cpu_available[closest_node] = diff_cpu
                                        w[fs, closest_node] = w[fs, closest_node] + y[f, i, fs, closest_node] * omega1 * m[f][fs]
                                        perc_used_cpu[closest_node] = perc_used_cpu[closest_node] + cpu_requested1 / node_cpu[closest_node]
                                        allocation_finished = True
                                    else:
                                        perc_cpu = self.node_cpu_available[closest_node] / cpu_requested
                                        self.node_cpu_available[closest_node] = 0.0
                                        y[f, i, fs, closest_node] = perc_cpu
                                        cpu_requested1 = cpu_requested * (1 - perc_cpu)
                                        w[fs, closest_node] = w[fs, closest_node] + y[f, i, fs, closest_node] * omega1 * m[f][fs]
                                        perc_used_cpu[closest_node] = perc_used_cpu[closest_node] + cpu_requested1 / node_cpu[
                                            closest_node]
                                    if not self.cfj[fs, closest_node]:
                                        self.node_memory_available[closest_node] = diff_memory
                                        self.cfj[fs, closest_node] = True
                            omega1 = 0
        return y, w, perc_used_cpu

    def fill_z(self, f, z, w, perc_used_cpu, perc_cpu_diff_usage):
        dag, functions, m, mf, nodes, ufj, node_cpu = self.basic_fill_data()
        parallel_successors_groups = dag.get_parallel_successors_indexes(functions[f])
        if len(parallel_successors_groups) > 0:
            for i in range(len(nodes)):
                omega = w[f, i]
                if omega > 0:
                    for par_group in parallel_successors_groups:
                        for fp in par_group:
                            omega1 = copy.deepcopy(omega)
                            for j in range(len(nodes)):
                                if omega1 > 0:
                                    cpu_requested = omega1 * m[f][fp] * ufj[fp, j]  # here z[f,i,fp,j] is 1
                                    memory_requested = mf[fp]  ## here z[f,i,fp,j] is 1
                                    cpu_requested1 = copy.deepcopy(cpu_requested)
                                    memory_requested1 = copy.deepcopy(memory_requested)
                                    allocation_finished = False
                                    while cpu_requested1 > 0:  # we can use cpu_requested or memory_requested
                                        if allocation_finished:
                                            break
                                        closest_node, _, _, f_placed = self.get_closest_available_node(f, fp, mf, i, perc_used_cpu, perc_cpu_diff_usage)
                                        if closest_node < 0:
                                            raise Exception(f'The nodes are overloaded, no more '
                                                            f'resources to be allocated! You are trying to allocate '
                                                            f'{cpu_requested1} cores in [{self.node_cpu_available}] and {memory_requested1}MB in [{self.node_memory_available}]')
                                        diff_cpu = self.node_cpu_available[closest_node] - cpu_requested1
                                        # get_closest_available_node returns node with enough memory
                                        diff_memory = self.node_memory_available[closest_node]
                                        if not f_placed:
                                            diff_memory = self.node_memory_available[closest_node] - memory_requested1  # get_closest_available_node
                                        if diff_cpu >= 0:
                                            if cpu_requested1 == cpu_requested:
                                                z[f, i, fp, closest_node] = 1.0
                                            else:
                                                z[f, i, fp, closest_node] = z[f, i, fp, closest_node] + cpu_requested1 / cpu_requested

                                            self.node_cpu_available[closest_node] = diff_cpu
                                            w[fp, closest_node] = w[fp, closest_node] + z[f, i, fp, closest_node] * omega1 * m[f][fp]
                                            perc_used_cpu[closest_node] = perc_used_cpu[closest_node] + cpu_requested1 / node_cpu[closest_node]
                                            allocation_finished = True
                                        else:
                                            perc_cpu = self.node_cpu_available[closest_node] / cpu_requested
                                            self.node_cpu_available[closest_node] = 0.0
                                            z[f, i, fp, closest_node] = perc_cpu
                                            cpu_requested1 = cpu_requested * (1 - perc_cpu)
                                            w[fp, closest_node] = w[fp, closest_node] + z[f, i, fp, closest_node] * omega1 * m[f][fp]
                                            perc_used_cpu[closest_node] = perc_used_cpu[closest_node] + cpu_requested1 /node_cpu[closest_node]

                                        if not self.cfj[fp, closest_node]:
                                            self.node_memory_available[closest_node] = diff_memory
                                            self.cfj[fp, closest_node] = True
                                    omega1 = 0
        return z, w, perc_used_cpu

    def heuristic_placement(self, perc_node_resources_x=0.2, perc_node_resources_y=0.2, perc_node_resources_z=0.2):
        start_time = time.time()
        _, functions, _, _, _, _, _ = self.basic_fill_data()
        x, w,  perc_used_cpu = self.fill_x(perc_node_resources_x)
        y = np.zeros((len(self.data.functions), len(self.data.nodes), len(self.data.functions), len(self.data.nodes)))
        z = np.zeros((len(self.data.functions), len(self.data.nodes), len(self.data.functions), len(self.data.nodes)))
        for f in range(len(functions)):
            y, w,  perc_used_cpu = self.fill_y(f, y, w, perc_used_cpu, perc_node_resources_y)
            z, w,  perc_used_cpu = self.fill_z(f, z, w, perc_used_cpu, perc_node_resources_z)
        end_time = time.time()
        self.set_coldstart()
        # Calculate the elapsed time
        decision_time = (end_time - start_time)*1000  # milliseconds
        return x, y, z, w, self.node_cpu_available, self.node_memory_available, self.cfj, decision_time

    def resource_usage(self):
        total_nodes = 0
        memory = 0
        cpus = 0
        memories = []
        cores = []
        nodes = len(self.cfj[0])
        functions = len(self.cfj)
        for i in range(nodes):
            memories.append(self.data.node_memories[i]-self.node_memory_available[i])
            cores.append(self.data.node_cores[i] - self.node_cpu_available[i])
            for f in range(functions):
                if self.cfj[f, i]:
                    total_nodes = total_nodes+1
                    break
        for m in memories:
            memory = memory+m
        for cpu in cores:
            cpus = cpus + cpu
        return total_nodes, memory, cpus

    def basic_fill_data(self):
        functions = self.get_functions()
        nodes = self.data.nodes
        dag = self.data.dag
        ufj = self.data.core_per_req_matrix
        mf = self.data.function_memories
        m = self.data.m
        node_cpu = self.data.node_cores
        return dag, functions, m, mf, nodes, ufj, node_cpu

    def as_instance(self, node):
        return self.node_memory_available[node]-self.data.node_memories[node] < 0

    def get_nodes(self, f):
        nodes =[]
        for j in range(len(self.data.nodes)):
            if self.cfj[f, j]:
                nodes.append(j)
        return nodes

    def set_coldstart(self):
        cold_starts = self.data.function_cold_starts
        max_coldstart = 0
        for f in range(len(cold_starts)):
            if cold_starts[f] > max_coldstart:
                max_coldstart = cold_starts[f]
        self.coldstart = max_coldstart


delays = np.zeros((7,2))
qty_values = 1
topology_position = 8   # select the topology
# app.set_topology(app, topology_position)
input = app.input
param.set_topology(param, topology_position, input)
f = len(input["function_names"])
nod = len(input["node_names"])
workload = param.workload_init(param, f, nod)

# Vary the workload from 50 to 300
# for i in range(7):
#     if i == 0:
#         lamb = 10
# else:
#     lamb = 50 * i
lamb = 10
workload[0][0] = lamb
json_workload = json.dumps(workload)
input["workload_on_source_matrix"] = json_workload

data = Data()
setup_community_data(data, input)
setup_runtime_data(data, input)
print(f'Nodes={data.nodes}')
perc_workload_balance = 1.0
asc = ASECO(data)

x, y, z, w, node_cpu_available, node_memory_available, instance_fj, decision_time = \
    asc.heuristic_placement( perc_workload_balance, perc_workload_balance, perc_workload_balance)
total_delay, cold_start, network_delay = asc.object_function_heuristic(w, x, y, z)
delays[0] = lamb, total_delay
np.set_printoptions(threshold=np.inf)
# print(f'x={x}')
print('++++++++++++++++++++++++++++++++++++++++++++++++')

# print(f'y={y}')

print('++++++++++++++++++++++++++++++++++++++++++++++++')

# print(f'z={z}')

print('+++++++++++++++++++++X+++++++++++++++++++++++++++')
for f in range(len(data.functions)):
    stop=False
    for fs in range(len(data.functions)):
        for i in range(len(data.nodes)):
            for j in range(len(data.nodes)):
                if z[f,i,fs,j]>0:
                    print(z[f,i,fs])
                    stop=True
                    break

print('+++++++++++++++++++++Y+++++++++++++++++++++++++++')
for f in range(len(data.functions)):
    stop=False
    for fs in range(len(data.functions)):
        for i in range(len(data.nodes)):
            for j in range(len(data.nodes)):
                if y[f,i,fs,j]>0:
                    print(y[f,i,fs])
                    stop=True
                    break

print(f'Y={y}')
print(f'w={w}')

print('++++++++++++++++++++++++++++++++++++++++++++++++')
print(f' instance_fj')
print(f'{instance_fj}')

print(f'Total Delay/ColdStart/Network delay/decisionTime={total_delay}/{cold_start}/{network_delay}/{decision_time}')
