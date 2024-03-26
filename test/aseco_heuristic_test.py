import copy
import json
import time

import numpy as np
from core import setup_runtime_data, setup_community_data, Data


node_cpu_usage = None
node_gpu_usage = None
node_memory_usage = None
# from simulations.json_file_hotel import JsonfileHotel as app
from simulations.json_file_sockshop import JsonfileSockshop as app
# from simulations.json_file_complex import JsonfileComplex as app
# from test.json_file_test_app import JsonfileAppTest as app
from core import Parameters as param

# input = app.input


class ASECO:

    # note SELECTED
    def get_functions(self, data):
        function = []
        for func in data.functions:
            function.append(func.split("/")[1])
        return function

    def object_function_heuristic(self, data,w,x, y, z):
        function = self.get_functions(data)
        qty_f = len(data.functions)
        qty_nodes = len(data.nodes)
        dag = data.dag
        m = data.m
        nrt = data.nrt
        network_delay = data.node_delay_matrix
        lamb = data.workload_on_source_matrix

        sum_f = 0
        for f in range(qty_f):
            sum_i = 0
            for i in range(qty_nodes):
                if lamb[f, i] > 0.1:
                    for j in range(qty_nodes):
                        sum_f = sum_f+network_delay[i][j] * x[f, i, j]*lamb[f, i]

                sum_sequential = 0
                for j in range(qty_nodes):
                    sum_fs = 0
                    seq_successor = dag.get_sequential_successors_indexes(function[f])
                    for fs in seq_successor:
                        # delay_y = y[f, i, fs, j]*m[f][fs]*w[f, i]*nrt[fs]
                        delay_y = y[f, i, fs, j] * m[f][fs] * w[f, i]

                        sum_fs = sum_fs + delay_y
                    if sum_fs * network_delay[i][j] > 0:
                        sum_sequential = sum_sequential + sum_fs * network_delay[i][j]
                        values_changed = 1
                parallel_successors_groups = dag.get_parallel_successors_indexes(function[f])
                sum_parallel = 0
                for par_group in parallel_successors_groups:
                    max_delay_z = float('-inf')
                    for fp in par_group:
                        for j in range(qty_nodes):
                            # delay_z = z[f, i, fp, j] * m[f][fp] * w[f, i] * nrt[fp] * network_delay[i][j]
                            delay_z = z[f, i, fp, j] * m[f][fp] * w[f, i] * network_delay[i][j]
                            if delay_z > max_delay_z:
                                max_delay_z = delay_z

                    if max_delay_z > float('-inf'):
                        sum_parallel = sum_parallel + max_delay_z
                sum_i = sum_i + sum_sequential + sum_parallel
            sum_f = sum_f + sum_i

        return sum_f


    # note SELECTED
    def get_closest_available_node(self, data, node_cpu_available, node_memory_available, f, mf, i, perc_used_cpu, perc_cpu_diff_usage):  # TODO
        selected_node = -1
        cpu = 0.0
        memory = 0.0
        min_delay = float('inf')
        nodes = data.nodes
        candidate_nodes = []
        less_used_nodes = []

        max_delay_f = data.max_delay_matrix
        node_delay = data.node_delay_matrix

        # prepare a list of nodes with available cores and memory
        for j in range(len(nodes)):
            if node_cpu_available[j] > 0 and node_memory_available[j] >= mf[f]:
                candidate_nodes.append(j)

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
                    # select the powerful node
                    candidates = []
                    for j in candidate_nodes:
                        if node_delay[i][selected_node] == node_delay[i][j]:
                            candidates.append(j)
                    powerful_node_capacity = 0
                    for node in candidates:
                        if node_cpu_available[node] > powerful_node_capacity:
                            powerful_node_capacity = node_cpu_available[node]
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
            cpu = node_cpu_available[selected_node]
            memory = node_memory_available[selected_node]
        return selected_node, cpu, memory


    # note SELECTED
    def fill_x(self, data, cpu_diff_usage=0.2):

        x = np.zeros((len(data.functions), len(data.nodes), len(data.nodes)))
        w = np.zeros((len(data.functions), len(data.nodes)))
        workload = data.workload_on_source_matrix
        node_cpu_available = copy.deepcopy(data.node_cores)
        node_memory_available = copy.deepcopy(data.node_memories)
        perc_used_cpu = np.zeros(len(data.nodes))
        instance_fj = np.zeros((len(data.functions), len(data.nodes)), dtype=bool)
        _, functions, _, mf, nodes, ufj, node_cpu = self.basic_fill_data(data)
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

                        closest_node, _, _ = self.get_closest_available_node(data, node_cpu_available, node_memory_available, f, mf, i, perc_used_cpu, cpu_diff_usage)

                        if closest_node < 0:
                            raise Exception("The nodes are overloaded, no more resources to be allocated!")
                        diff_cpu = node_cpu_available[closest_node] - cpu_requested1
                        diff_memory = node_memory_available[closest_node] - memory_requested1  # get_closest_available_node
                        # returns node with enough memory
                        if diff_cpu >= 0:
                            if cpu_requested1 == cpu_requested:
                                x[f, i, closest_node] = 1.0
                            else:
                                x[f, i, closest_node] = cpu_requested1/cpu_requested  # x[f, i, closest_node] +

                            node_cpu_available[closest_node] = diff_cpu
                            w[f, closest_node] = w[f, closest_node] + x[f, i, closest_node] * lamb
                            perc_used_cpu[closest_node] = perc_used_cpu[closest_node] + cpu_requested1 / node_cpu[
                                closest_node]
                            allocation_finished = True
                        else:
                            perc_cpu = node_cpu_available[closest_node] / cpu_requested
                            node_cpu_available[closest_node] = 0.0
                            x[f, i, closest_node] = perc_cpu
                            cpu_requested1 = cpu_requested * (1 - perc_cpu)
                            w[f, closest_node] = w[f, closest_node] + x[f, i, closest_node] * lamb
                            perc_used_cpu[closest_node] = perc_used_cpu[closest_node] + cpu_requested1 / node_cpu[
                                closest_node]
                        if not instance_fj[f, closest_node]:
                            node_memory_available[closest_node] = diff_memory
                            instance_fj[f, closest_node] = True
        # print(f'W-x={w}')
        return x, w, node_cpu_available, node_memory_available, perc_used_cpu, instance_fj

    def fill_y(self, data, f, y, w, node_cpu_available, node_memory_available, perc_used_cpu, perc_cpu_diff_usage, instance_fj):
        dag, functions, m, mf, nodes, ufj, node_cpu = self.basic_fill_data(data)
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
                                    closest_node, _, _ = self.get_closest_available_node(data, node_cpu_available, node_memory_available, f, mf, i, perc_used_cpu, perc_cpu_diff_usage)
                                    if closest_node < 0:
                                        raise Exception("The nodes are overloaded, no more resources to be allocated!")
                                    diff_cpu = node_cpu_available[closest_node] - cpu_requested1
                                    # get_closest_available_node returns node with enough memory
                                    diff_memory = node_memory_available[closest_node] - memory_requested
                                    if diff_cpu >= 0:
                                        if cpu_requested1 == cpu_requested:
                                            y[f, i, fs, closest_node] = 1.0
                                        else:
                                            y[f, i, fs, closest_node] = cpu_requested1 / cpu_requested # y[f, i, fs, closest_node] +

                                        node_cpu_available[closest_node] = diff_cpu
                                        w[fs, closest_node] = w[fs, closest_node] + y[f, i, fs, closest_node] * omega1 * m[f][fs]
                                        perc_used_cpu[closest_node] = perc_used_cpu[closest_node] + cpu_requested1 / node_cpu[closest_node]
                                        allocation_finished = True
                                    else:
                                        perc_cpu = node_cpu_available[closest_node] / cpu_requested
                                        node_cpu_available[closest_node] = 0.0
                                        y[f, i, fs, closest_node] = perc_cpu
                                        cpu_requested1 = cpu_requested * (1 - perc_cpu)
                                        w[fs, closest_node] = w[fs, closest_node] + y[f, i, fs, closest_node] * omega1 * m[f][fs]
                                        perc_used_cpu[closest_node] = perc_used_cpu[closest_node] + cpu_requested1 / node_cpu[
                                            closest_node]
                                    if not instance_fj[fs, closest_node]:
                                        node_memory_available[closest_node] = diff_memory
                                        instance_fj[fs, closest_node] = True
                            omega1 = 0
        return y, w, node_cpu_available, node_memory_available, perc_used_cpu, instance_fj

    def fill_z(self, data, f, z, w, node_cpu_available, node_memory_available, perc_used_cpu, perc_cpu_diff_usage, instance_fj):
        dag, functions, m, mf, nodes, ufj, node_cpu = self.basic_fill_data(data)
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
                                        closest_node, _, _ = self.get_closest_available_node(data, node_cpu_available, node_memory_available,
                                                                                        f, mf, i, perc_used_cpu, perc_cpu_diff_usage)
                                        if closest_node < 0:
                                            raise Exception(f'The nodes are overloaded, no more '
                                                            f'resources to be allocated! You are trying to allocate '
                                                            f'{cpu_requested1} cores in [{node_cpu_available}] and {memory_requested1}MB in [{node_memory_available}]')
                                        diff_cpu = node_cpu_available[closest_node] - cpu_requested1
                                        # get_closest_available_node returns node with enough memory
                                        diff_memory = node_memory_available[closest_node] - memory_requested1
                                        if diff_cpu >= 0:
                                            if cpu_requested1 == cpu_requested:
                                                z[f, i, fp, closest_node] = 1.0
                                            else:
                                                z[f, i, fp, closest_node] = z[f, i, fp, closest_node] + cpu_requested1 / cpu_requested

                                            node_cpu_available[closest_node] = diff_cpu
                                            w[fp, closest_node] = w[fp, closest_node] + z[f, i, fp, closest_node] * omega1 * m[f][fp]
                                            perc_used_cpu[closest_node] = perc_used_cpu[closest_node] + cpu_requested1 / node_cpu[closest_node]
                                            allocation_finished = True
                                        else:
                                            perc_cpu = node_cpu_available[closest_node] / cpu_requested
                                            node_cpu_available[closest_node] = 0.0
                                            z[f, i, fp, closest_node] = perc_cpu
                                            cpu_requested1 = cpu_requested * (1 - perc_cpu)
                                            w[fp, closest_node] = w[fp, closest_node] + z[f, i, fp, closest_node] * omega1 * m[f][fp]
                                            perc_used_cpu[closest_node] = perc_used_cpu[closest_node] + cpu_requested1 / node_cpu[closest_node]

                                        if not instance_fj[fp, closest_node]:
                                            node_memory_available[closest_node] = diff_memory
                                            instance_fj[fp, closest_node] = True
                                    omega1 = 0
        return z, w, node_cpu_available, node_memory_available, perc_used_cpu, instance_fj

    def heuristic_placement(self, data, perc_node_resources_x=0.2, perc_node_resources_y=0.2, perc_node_resources_z=0.2):
        _, functions, _, _, _, _, _ = self.basic_fill_data(data)
        x, w, node_cpu_available, node_memory_available, perc_used_cpu, instance_fj = self.fill_x(data, perc_node_resources_x)
        y = np.zeros((len(data.functions), len(data.nodes), len(data.functions), len(data.nodes)))
        z = np.zeros((len(data.functions), len(data.nodes), len(data.functions), len(data.nodes)))
        for f in range(len(functions)):
            y, w, node_cpu_available, node_memory_available, perc_used_cpu, instance_fj = \
                self.fill_y(data, f, y, w, node_cpu_available, node_memory_available, perc_used_cpu,
                            perc_node_resources_y, instance_fj)
            z, w, node_cpu_available, node_memory_available, perc_used_cpu, instance_fj =\
                self.fill_z(data, f, z, w, node_cpu_available, node_memory_available, perc_used_cpu,
                            perc_node_resources_z, instance_fj)
        return x, y, z, w, node_cpu_available, node_memory_available, instance_fj

    def resource_usage(self, data, cfj, node_memory_available, node_cpu_available):
        total_nodes = 0
        memory = 0
        cpus = 0
        memories = []
        cores = []
        nodes = len(cfj[0])
        functions = len(cfj)
        for i in range(nodes):
            memories.append(data.node_memories[i]-node_memory_available[i])
            cores.append(data.node_cores[i] - node_cpu_available[i])
            for f in range(functions):
                if cfj[f, i] == 1:
                    total_nodes = total_nodes+1
                    break
        for m in memories:
            memory = memory+m
        for cpu in cores:
            cpus = cpus + cpu
        return total_nodes, memory, cpus

    def basic_fill_data(self, data):
        functions = self.get_functions(data)
        nodes = data.nodes
        dag = data.dag
        ufj = data.core_per_req_matrix
        mf = data.function_memories
        m = data.m
        node_cpu = data.node_cores
        return dag, functions, m, mf, nodes, ufj, node_cpu

network_delays = np.zeros((1, 2))
total_delays = np.zeros((1,2))
qty_values = 1
topology_position = 7   # select the topology
# app.set_topology(app, topology_position)
input = app.input
param.set_topology(param, topology_position, input)
f = len(input["function_names"])
nod = len(input["node_names"])
workload = param.workload_init(param, f, nod)

lamb = 200
# Vary the workload from 50 to 300
# for i in range(2):
#     asc = ASECO()
#     if i == 0:
#         lamb = 10
#     else:
#         lamb = 50 * i
asc = ASECO()
workload[0][0] = lamb
json_workload = json.dumps(workload)
input["workload_on_source_matrix"] = json_workload

data = Data()
setup_community_data(data, input)
setup_runtime_data(data, input)
print(f'Nodes={data.nodes}')
perc_workload_balance = 1.0

start_time = time.time()
x, y, z, w, node_cpu_available, node_memory_available, instance_fj = \
    asc.heuristic_placement(data, perc_workload_balance, perc_workload_balance, perc_workload_balance)
end_time = time.time()

# Calculate the elapsed time
elapsed_time = end_time - start_time
network_delays[0] = lamb, asc.object_function_heuristic(data, w, x, y, z)
total_delays[0] = lamb, round(asc.object_function_heuristic(data, w, x, y, z)+elapsed_time*1000)
total_nodes, memory, cores = asc.resource_usage(data, instance_fj, node_memory_available, node_cpu_available)

np.set_printoptions(threshold=np.inf, linewidth=np.inf)
print(f'x={x}')
print('++++++++++++++++++++++++++++++++++++++++++++++++')

print(f'y={y}')

print('++++++++++++++++++++++++++++++++++++++++++++++++')

print(f'z={z}')

print('++++++++++++++++++++++++++++++++++++++++++++++++')

print(f'w={w}')

print('++++++++++++++++++++++++++++++++++++++++++++++++')
print(f' instance_fj')
print(f'{instance_fj}')
print(f'Processing delay={elapsed_time} s')
print(f'DELAY (without processing time)={network_delays}ms')
print(f'DELAY (with processing time)={total_delays} ms')
print(f'TOTAL NODE={total_nodes}')
print(f'MEMORY USED={memory}MB')
print(f'CORES={cores} Millicores')
