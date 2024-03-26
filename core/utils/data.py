from typing import List
import numpy as np


class Data:

    def __init__(self, sources: List[str] = None, nodes: List[str] = None, functions: List[str] = None, dag=None,
                 m=None, parallel_scheduler=None, node_cores: List[str] = None, nrt=None,
                 function_memories=None, node_memories=None, function_cold_starts=None):
        # Convert Python lists to NumPy arrays
        self.sources = np.array(sources) if sources else np.empty(0, dtype=np.object_)
        self.nodes = np.array(nodes) if nodes else np.empty(0, dtype=np.object_)
        self.functions = np.array(functions) if functions else np.empty(0, dtype=np.object_)
        self.m = np.array(m) if m is not None else np.empty((0, 0), dtype=np.float64)
        self.parallel_scheduler = np.array(parallel_scheduler) if parallel_scheduler is not None else np.empty((0, 0), dtype=np.float64)
        self.nrt = np.array(nrt) if nrt is not None else np.empty((0, 0), dtype=np.float64)
        self.function_memories = np.array(function_memories) if function_memories else np.empty(0, dtype=np.float64)
        self.node_memories = np.array(node_memories) if node_memories else np.empty(0, dtype=np.float64)
        self.function_cold_starts = np.array(function_cold_starts) if function_cold_starts else np.empty(0, dtype=np.float64)



        # Initialize other arrays with appropriate types
        self.node_memory_matrix: np.array = np.empty((0, 0), dtype=np.float64)
        self.function_memory_matrix: np.array = np.empty((0, 0), dtype=np.float64)
        self.node_delay_matrix: np.array = np.empty((0, 0), dtype=np.float64)
        self.workload_matrix: np.array = np.empty((0, 0), dtype=np.float64)
        self.workload_on_source_matrix: np.array = np.empty((0, 0), dtype=np.float64)
        self.max_delay_matrix: np.array = np.empty((0, 0), dtype=np.float64)
        self.response_time_matrix: np.array = np.empty((0, 0), dtype=np.float64)
        self.node_cores_matrix: np.array = np.empty((0, 0), dtype=np.float64) # NOTE: do we need this?
        self.node_cores = np.array(node_cores) if node_cores else np.empty(0, dtype=np.float64)
        self.cores_matrix: np.array = np.empty((0, 0), dtype=np.float64)
        self.actual_cpu_allocations = np.empty(0, dtype=np.float64)
        self.dag = dag


        self.old_allocations_matrix: np.array = np.empty((0, 0), dtype=np.float64)
        self.core_per_req_matrix: np.array = np.empty((0, 0), dtype=np.float64)

        self.gpu_function_memory_matrix: np.array = np.empty((0, 0), dtype=np.float64)
        self.gpu_node_memory_matrix: np.array = np.empty((0, 0), dtype=np.float64)
        self.prev_x: np.array = np.empty(0, dtype=np.float64)

        self.node_costs: np.array = np.empty((0, 0), dtype=np.float64)
        self.node_budget: int = 0
