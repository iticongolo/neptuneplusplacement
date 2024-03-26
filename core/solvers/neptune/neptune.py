import time

from ..solver import Solver
from .neptune_step1 import *
from .neptune_step2 import *
from .utils.output import convert_x_matrix, convert_c_matrix
from core.utils.aseco_heuristic import ASECO

class NeptuneBase(Solver):
    def __init__(self, step1=None, step2_delete=None, step2_create=None, **kwargs):
        super().__init__(**kwargs)
        self.step1 = step1
        self.step2_delete = step2_delete
        self.step2_create = step2_create
        self.solved = False
        self.computing_time = 0

    def init_vars(self): pass
    def init_constraints(self): pass

    def solve(self):
        start_time = time.time()
        self.step1.load_data(self.data)
        self.step1.solve()
        self.step1_x, self.step1_c = self.step1.results()
        self.data.max_score = self.step1.score()
        self.step2_delete.load_data(self.data)
        self.solved = self.step2_delete_solved = self.step2_delete.solve()
        self.step2_x, self.step2_c = self.step2_delete.results()
        if not self.solved:
           self.step2_create.load_data(self.data)
           self.solved = self.step2_create.solve()
           self.step2_x, self.step2_c = self.step2_create.results()
        end_time = time.time()
        self.computing_time = self.computing_time+end_time-start_time
        return self.solved
    
    def results(self): 
        if self.solved:
            return convert_x_matrix(self.step2_x, self.data.nodes, self.data.functions, self.data.nodes), convert_c_matrix(self.step2_c, self.data.functions, self.data.nodes)
        else:
            return convert_x_matrix(self.step1_x, self.data.nodes, self.data.functions, self.data.nodes), convert_c_matrix(self.step1_c, self.data.functions, self.data.nodes)
          
    def score(self):
        return {"step1 (Network delay)": self.step1.score(), "step2 (added-nodes)": self.step2_delete.score() if self.step2_delete_solved else self.step2_create.score() }

    def get_functions(self, data):
        function = []
        for func in data.functions:
            function.append(func.split("/")[1])
        return function

    # return the network delay from node i to the function f2
    # the delay can be maximum or minimum according to the boolean is_closest
    def getNode(self, node_i, f2, cfj, is_closest=True):
        selected_delay = float('inf')
        node_j = node_i
        delayij = self.data.node_delay_matrix
        nodes=delayij.shape[0]
        if is_closest:
            for j in range(nodes):
                if cfj[f2,j]>0:
                    if selected_delay>delayij[node_i, j]:
                        selected_delay=delayij[node_i, j]
                        node_j=j
        else:
            selected_delay=0
            for j in range(nodes):
                if cfj[f2,j]>0:
                    if selected_delay<delayij[node_i, j]:
                        selected_delay=delayij[node_i, j]
                        node_j=j
        return node_j

    def dep_results(self):
        start_time = time.time()
        funcs, function, nodes = self.basic_data()
        x_neptune = self.step2_x
        cfj = self.step2_c
        x=np.zeros(x_neptune.shape)
        y= np.zeros((funcs, nodes, funcs, nodes))
        z = np.zeros((funcs, nodes, funcs, nodes))
        dag = self.data.dag
        workload = self.data.workload_matrix
        # fill x[f,i,j]
        for f in range(funcs):
            for i in range(nodes):
                if workload[f,i] > 0.0000000001:
                    x[f,i] = x_neptune[f,i]

        # fill y[f,i,g,j]
        for f in range(funcs):
            for i in range(nodes):
                seq_successor = dag.get_sequential_successors_indexes(function[f])
                for fs in seq_successor:
                    if cfj[f, i]:
                        j = self.getNode(i, fs, cfj)
                        y[f, i, fs, j] = 1

         # fill z[f,i,g,j]
                parallel_successors_groups = dag.get_parallel_successors_indexes(function[f])
                for par_group in parallel_successors_groups:
                    for fp in par_group:
                        if cfj[f, i]:
                            j = self.getNode(i, fp, cfj)
                            z[f, i, fp, j] = 1
        end_time = time.time()
        self.computing_time = self.computing_time+end_time-start_time
        return x, y, z

    def basic_data(self):
        function = self.get_functions(self.data)
        nodes = len(self.data.nodes)
        funcs = len(self.data.functions)
        return funcs, function, nodes
    def intenal_workload(self):
        funcs, function, nodes = self.basic_data()
        w = np.zeros((funcs, nodes))
        x, y, z = self.dep_results()
        lamb = self.data.workload_matrix
        dag = self.data.dag
        m = self.data.m

        # partial workload x*lamb for direct requests
        for f in range(funcs):
            for i in range(nodes):
                for j in range(nodes):
                    w[f, j] = w[f, j] + x[f, i, j]*lamb[f, i]

        # partial internal and external workload for sequential invocations
        for f in range(funcs):
            for i in range(nodes):
                for j in range(nodes):
                    seq_successor = dag.get_sequential_successors_indexes(function[f])
                    for fs in seq_successor:
                        w[fs, j] = w[fs, j] + y[f, i, fs, j] * m[f][fs] * w[f, i]

        # partial internal and external workload for parallel invocations
        for f in range(funcs):
            for i in range(nodes):
                for j in range(nodes):
                    parallel_successors_groups = dag.get_parallel_successors_indexes(function[f])
                    for par_group in parallel_successors_groups:
                        for fp in par_group:
                            w[fp, j] = w[fp, j] + z[f, i, fp, j] * m[f][fp] * w[f, i]
        return w

    def dep_networkdelay(self):
        w = self.intenal_workload()
        # print(f'W={w}')
        funcs, function, nodes = self.basic_data()
        dag = self.data.dag
        m = self.data.m
        network_delay = self.data.node_delay_matrix

        lamb = self.data.workload_matrix
        x, y, z = self.dep_results()

        sum_f = 0
        for f in range(funcs):
            sum_i = 0
            for i in range(nodes):
                if lamb[f, i] > 0.1:
                    for j in range(nodes):
                        sum_f = sum_f+network_delay[i][j] * x[f, i, j]*lamb[f, i]

                sum_sequential = 0
                for j in range(nodes):
                    sum_fs = 0
                    seq_successor = dag.get_sequential_successors_indexes(function[f])
                    for fs in seq_successor:
                        # delay_y = y[f, i, fs, j]*m[f][fs]*w[f, i]*nrt[fs]
                        delay_y = y[f, i, fs, j] * m[f][fs] * w[f, i]

                        sum_fs = sum_fs + delay_y
                    sum_sequential = sum_sequential + sum_fs * network_delay[i][j]
                parallel_successors_groups = dag.get_parallel_successors_indexes(function[f])
                sum_parallel = 0
                for par_group in parallel_successors_groups:
                    max_delay_z = float('-inf')
                    for fp in par_group:
                        for j in range(nodes):
                            # delay_z = z[f, i, fp, j] * m[f][fp] * w[f, i] * nrt[fp] * network_delay[i][j]
                            delay_z = z[f, i, fp, j] * m[f][fp] * w[f, i] * network_delay[i][j]
                            if delay_z > max_delay_z:
                                max_delay_z = delay_z

                    if max_delay_z > float('-inf'):
                        sum_parallel = sum_parallel + max_delay_z
                sum_i = sum_i + sum_sequential + sum_parallel
            sum_f = sum_f + sum_i

        return sum_f

    def object_function_global_results(self):
        aseco = ASECO(self.data)
        x, y, z = self.dep_results()
        w = self.intenal_workload()
        lamb = self.data.workload_matrix
        aseco.set_coldstart()
        total_delay, coldstart, network_delay = aseco.object_function_heuristic(w, x, y, z, lambd=lamb,
                                                                                instances=self.step2_c)
        decision_time = self.computing_time*1000  # in milliseconds
        return total_delay, coldstart, network_delay, decision_time

    def get_resources_usage(self):
        nodes = []
        instancef = []
        for f in range(len(self.step2_c)):
            instances = 0
            for j in range(len(self.step2_c[0])):
                if self.step2_c[f, j]:
                    instances = instances+self.step2_c[f, j]
                    if j not in nodes:
                        nodes.append(j)
            instancef.append([f, instances])

        return len(nodes), instancef

    def get_memory_used(self, mf):
        _, instances = self.get_resources_usage()
        memory = 0
        for f in range(len(instances)):
            memory = memory + instances[f][1]*mf[f]
        return memory

    # def get_total_cold_starts(self, coldstart_f):
    #     _, instances = self.get_resources_usage()
    #     cold_starts = 0
    #     for f in range(len(instances)):
    #         cold_starts = cold_starts + instances[f][1]*coldstart_f[f]
    #     return cold_starts


class NeptuneMinDelayAndUtilization(NeptuneBase):
    def __init__(self, **kwargs):
        super().__init__(
            NeptuneStep1CPUMinDelayAndUtilization(**kwargs), 
            NeptuneStep2MinDelayAndUtilization(mode="delete", **kwargs),
            NeptuneStep2MinDelayAndUtilization(mode="create", **kwargs),
            **kwargs
            )


class NeptuneMinDelay(NeptuneBase):
    def __init__(self, **kwargs):
        super().__init__(
            NeptuneStep1CPUMinDelay(**kwargs), 
            NeptuneStep2MinDelay(mode="delete", **kwargs),
            NeptuneStep2MinDelay(mode="create", **kwargs),
            **kwargs
            )

class NeptuneMinUtilization(NeptuneBase):
    def __init__(self, **kwargs):
        super().__init__(
            NeptuneStep1CPUMinUtilization(**kwargs), 
            NeptuneStep2MinUtilization(mode="delete", **kwargs),
            NeptuneStep2MinUtilization(mode="create", **kwargs),
            **kwargs
            )
