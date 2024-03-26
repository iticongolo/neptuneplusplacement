import copy
import json

import numpy as np
import time
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


class HeuXu:
    def __init__(self, data):
        self.cfj = np.zeros((len(data.functions), len(data.nodes)))
        self.node_cpu_available = copy.deepcopy(data.node_cores)
        self.node_memory_available = copy.deepcopy(data.node_memories)
        self.data = data
        self.list_r_cfj = []
        self.w = []
        self.x = []
        self.y = []
        self.z = []

    def fill_w(self, request):  # will fill internal workload for each single request
        dag, functions, m, mf, nodes, ufj, node_cpu = self.basic_fill_data()
        lamb = copy.deepcopy(self.data.workload_on_source_matrix[request])
        for f in range(len(functions)):
            for i in range(len(nodes)):
                if lamb[f, i] > 0.001:
                    seq_successor = dag.get_sequential_successors_indexes(functions[f])
                    if len(seq_successor) > 0:
                        for fs in seq_successor:
                            lamb[fs, i] = lamb[fs, i] + lamb[f, i] * m[f][fs]
                    group_par_successor = dag.get_parallel_successors_indexes(functions[f])
                    if len(group_par_successor) > 0:
                        for group in group_par_successor:
                            for fp in group:
                                lamb[fp, i] = lamb[fp, i] + lamb[f, i] * m[f][fp]
        self.w.append(lamb)

    def basic_fill_data(self):
        functions = self.get_functions(self.data)
        nodes = self.data.nodes
        dag = self.data.dag
        ufj = self.data.core_per_req_matrix
        mf = self.data.function_memories
        m = self.data.m
        node_cpu = self.data.node_cores
        return dag, functions, m, mf, nodes, ufj, node_cpu

    # first step
    def place_app(self, request, i,  parallel_scheduler):
        r_cfj = np.zeros((len(self.data.functions), len(self.data.nodes)))
        for layer in parallel_scheduler:
            memory_required = self.get_memory(layer)
            cpu_required = self.get_cpu_demand(layer, request, i)
            j, remain_cpu, remain_memory = self.get_closest_available_node(self.node_cpu_available,
                                                                           self.node_memory_available, layer,
                                                                           memory_required, cpu_required, i, self.cfj)
            if j < 0:
                raise Exception(f'The nodes are overloaded, no more '
                                f'resources to be allocated! You are trying to allocate '
                                f'{cpu_required} millicores in [{self.node_cpu_available}] and {memory_required}MB in '
                                f'[{self.node_memory_available}]')
            for f in layer:
                r_cfj[f, j] = 1
                self.cfj[f, j] = 1
            self.node_cpu_available[j] = remain_cpu
            self.node_memory_available[j] = remain_memory
        self.list_r_cfj.append(r_cfj)

    # note SELECTED
    def get_functions(self, data):
        function = []
        for func in data.functions:
            function.append(func.split("/")[1])
        return function

    def object_function_heu(self, request):
        dag, functions, m, mf, nodes, ufj, node_cpu = self.basic_fill_data()
        lamb = copy.deepcopy(self.data.workload_on_source_matrix[request])
        function = self.get_functions(self.data)
        qty_f = len(self.data.functions)
        qty_nodes = len(self.data.nodes)
        dag = self.data.dag
        network_delay = self.data.node_delay_matrix
        x = self.x[request]
        y = self.y[request]
        z = self.z[request]
        w = self.w[request]

        sum_f = 0
        for f in range(qty_f):
            sum_i = 0
            for i in range(qty_nodes):
                if lamb[f, i] > 0.001:
                    for j in range(qty_nodes):
                        sum_f = sum_f + network_delay[i][j] * x[f, i, j] * lamb[f, i]

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

    # note: SELECTED
    # we assume node as a cloudlet to be aligned with the assumption of Xu et. al (2023)
    # We consider available only the nodes with enough capacity to place all the
    # functions in a given layer-list_f ( independent functions)
    def get_closest_available_node(self, node_cpu_available, node_memory_available,
                                   list_f, m_list_f, cpu_demand,  i, cfj):
        selected_node = -1
        cpu = 0.0
        memory = 0.0
        min_delay = float('inf')
        nodes = self.data.nodes
        candidate_nodes = []
        node_delay = self.data.node_delay_matrix
        # print(node_memory_available)
        # prepare a list of nodes with available cores and memory
        for j in range(len(nodes)):
            if node_cpu_available[j] >= cpu_demand:
                for f in list_f:
                    if cfj[f, j] or node_memory_available[j] >= m_list_f:
                        candidate_nodes.append(j)
        if i in candidate_nodes:
            selected_node = i
        else:
            for j in candidate_nodes:
                if min_delay > node_delay[i][j]:
                    min_delay = node_delay[i][j]
                    selected_node = j


        if selected_node >= 0:
            cpu = node_cpu_available[selected_node]-cpu_demand
            memory = node_memory_available[selected_node]
            if len(list_f) > 0:
                f = list_f[0]
                if not cfj[f, selected_node]:
                    memory = memory - m_list_f
        return selected_node, cpu, memory

    def fill_x(self, functions, nodes, cfj_r, lamb_r):
        x_r = np.zeros((len(functions), len(nodes), len(nodes)))
        for f in range(len(functions)):
            for i in range(len(nodes)):
                if lamb_r[f, i] > 0.001:
                    for j in range(len(nodes)):
                        if cfj_r[f, j]:
                            x_r[f, i, j] = 1
        self.x.append(x_r)

    def fill_y(self, dag, functions, nodes, cfj_r):
        y_r = np.zeros((len(functions), len(nodes), len(functions), len(nodes)))
        for f in range(len(functions)):
            seq_successor = dag.get_sequential_successors_indexes(functions[f])
            if len(seq_successor) > 0:
                for i in range(len(nodes)):
                    if cfj_r[f, i]:
                        for fs in seq_successor:
                            for j in range(len(nodes)):
                                if cfj_r[fs, j]:
                                    y_r[f, i, fs, j] = 1
        self.y.append(y_r)

    def fill_z(self, dag, functions, nodes, cfj_r):
        z_r = np.zeros((len(functions), len(nodes), len(functions), len(nodes)))
        for f in range(len(functions)):
            par_successor = dag.get_parallel_successors_indexes(functions[f])
            if len(par_successor) > 0:
                for i in range(len(nodes)):
                    if cfj_r[f, i]:
                        for group in par_successor:
                            for fp in group:
                                for j in range(len(nodes)):
                                    if cfj_r[fp, j]:
                                        z_r[f, i, fp, j] = 1
        self.z.append(z_r)

    def fill_xyz(self, request):
        dag, functions, _, _, nodes, _, _ = self.basic_fill_data()
        cfj_r = self.list_r_cfj[request]
        lamb_r = copy.deepcopy(self.data.workload_on_source_matrix[request])
        # fill x
        self.fill_x(functions, nodes, cfj_r, lamb_r)

        # fill y
        self.fill_y(dag, functions, nodes, cfj_r)

        # fill z
        self.fill_z(dag, functions, nodes, cfj_r)

    def get_memory(self, layer):
        total_memory = 0
        _, _, _, mf, _, _, _ = self.basic_fill_data()
        for f in layer:
            total_memory = total_memory + mf[f]
        return total_memory

    def get_cpu_demand(self, layer, request, i):
        total_cpu = 0

        _, _, _, _, _, ufj, _ = self.basic_fill_data()
        for f in layer:
            total_cpu = total_cpu + self.w[request][f][i]*ufj[f, i]
        return total_cpu

    def resource_usage(self):
        total_nodes = 0
        memory = 0
        cpus = 0
        memories=[]
        cores=[]
        nodes = len(self.cfj[0])
        functions = len(self.cfj)
        for i in range(nodes):
            memories.append(self.data.node_memories[i]-self.node_memory_available[i])
            cores.append(self.data.node_cores[i] - self.node_cpu_available[i])
            for f in range(functions):
                if self.cfj[f, i] == 1:
                    total_nodes = total_nodes+1
                    break
        for m in memories:
            memory = memory+m
        for cpu in cores:
            cpus = cpus + cpu
        return total_nodes, memory, cpus

    def heuristic_placement(self, request, i, parallel_scheduler):
        start_time1 = time.time()
        self.fill_w(request)
        self.place_app(request, i, parallel_scheduler)
        end_time1 = time.time()
        t2 = end_time1 - start_time1
        print(f'T2={t2}')

        start_time3 = time.time()
        self.fill_xyz(request)
        end_time3 = time.time()
        t3 = end_time3 - start_time3
        print(f'T3={t3}')

        return t2, t3, self.x[request], self.y[request], self.z[request], self.w[request],  self.list_r_cfj, self.cfj

delays = np.zeros((1,2))
requests = 200
qty_values = 1
lamb_f = 1
workload = []
topology_position = 7   # select the topology
# app.set_topology(app, topology_position)
input = app.input
param.set_topology(param, topology_position, input)
f = len(input["function_names"])
nod = len(input["node_names"])
workload_r = param.workload_init(param, f, nod)
workload_r[0][0] = lamb_f
for i in range(requests):
    workload.append(workload_r)
json_workload = json.dumps(workload)
input["workload_on_source_matrix"] = json_workload

data = Data()
setup_community_data(data, input)
setup_runtime_data(data, input)
parallel_scheduler = data.parallel_scheduler
asc = HeuXu(data)
delay = 0
delay1 = 0
cfj = {}
elapsed_time1 =0
end_time1=0
start_time1=0
start_time = time.time()
for request in range(requests):
    t2, t3, x, y, z, w, list_cfj, cfj = asc.heuristic_placement(request, 0, parallel_scheduler)
    # start_time1 = time.time()
    # network_delay = asc.object_function_heu(request)
    # end_time1 = time.time()
    delay = delay + t2
    delay1 = delay1 + t3
end_time = time.time()

# Calculate the elapsed time
elapsed_time = ((end_time - start_time))
elapsed_time1 = end_time1 - start_time1

print("Elapsed time:", elapsed_time, "ms")
total_nodes, memory, cores = asc.resource_usage()
print(f'Processing delay={elapsed_time1} s')
print(f'DELAYS (withount processing time)={delay}')
print(f'DELAYS1 (withount processing time)={delay1}')
print(f'DELAYS (with processing time)={round(delay+elapsed_time)}')
print(f'total_nodes={total_nodes}')
print(f'memory={memory}')
print(f'cores={cores}')

start_time = time.time()
end_time = time.time()