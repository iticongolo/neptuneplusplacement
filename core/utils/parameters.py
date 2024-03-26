from .topology import nodes, network

class Parameters:
    def workload_init(self, f, nod):
        workload = []
        for f in range(f):
            row_workload = []
            for node in range(nod):
                row_workload.append(0)
            workload.append(row_workload)
        return workload

    def set_topology(self, pos, app_input):
        # Update cpu_allocation dictionary after changing position
        cpu_allocation = {}
        for function in app_input["function_names"]:
            cpu_allocation[function] = {}
            for node in nodes[pos]:
                cpu_allocation[function][node] = True

        app_input["actual_cpu_allocations"] = cpu_allocation
        app_input["node_names"] = nodes[pos]
        app_input["node_delay_matrix"] = network[pos]['node_delay_matrix']
        app_input["node_memories"] = network[pos]["node_memories"]
        app_input["node_cores"] = network[pos]["node_cores"]
