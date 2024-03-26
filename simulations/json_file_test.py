from core import nodes, network


class JsonfileTest:

    qty_f = 4
    cpu_allocation = {}
    position = 0

    functions = ["ns/f0", "ns/f1", "ns/f2", "ns/f3"]

    input = {
        "with_db": False,
        "solver": {
            "type": "NeptuneMinDelay",
            "args": {"alpha": 0.0, "verbose": True}
        },
        "cpu_coeff": 1,
        "community": "community-test",
        "namespace": "namespace-test",
        "node_names": nodes[position],
        "node_delay_matrix": network[position]['node_delay_matrix'],

        "workload_on_source_matrix": [],

        "node_memories": network[position]["node_memories"],

        "execution_time": [
            78, 13, 33, 16
        ],
        "node_cores": network[position]["node_cores"],

        "gpu_node_names": [
        ],
        "gpu_node_memories": [
        ],
        "function_names": functions,
        "function_memories": [
            1000, 1000, 500, 500
        ],
        "function_cold_starts": [
            500, 500, 500, 500
        ],
        "function_max_delays": [
            5, 5, 5, 5
        ],
        "gpu_function_names": [
        ],
        "gpu_function_memories": [
        ],
        "actual_cpu_allocations": cpu_allocation,
        "app": "hotel",
        "actual_gpu_allocations": {
        },
        "nodes": [
            {"name": "f0", "users": 0, "nrt": 10},
            {"name": "f1", "users": 2, "nrt": 15},
            {"name": "f2", "users": 2, "nrt": 10},
            {"name": "f3", "users": 2, "nrt": 10},
        ],
        "edges": [
            {"source": "f0", "target": "f1", "sync": 1, "times": 2},
            {"source": "f1", "target": "f2", "sync": 2, "times": 3},
            {"source": "f1", "target": "f3", "sync": 2, "times": 5},
        ],
        "m": [[0, 2, 0, 0], [0, 0, 3, 5], [0, 0, 0, 0], [0, 0, 0, 0]],
        "parallel_scheduler": [[0], [1], [2, 3]]
    }
