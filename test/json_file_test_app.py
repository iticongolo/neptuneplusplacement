from core import nodes, network


class JsonfileAppTest:

    qty_f = 5
    cpu_allocation = {}
    position = 8

    functions = ["ns/f1", "ns/f2", "ns/f3", "ns/f4", "ns/f5"]

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

        "nrt": [
            200, 50, 20, 10, 10
        ],
        "node_cores": network[position]["node_cores"],

        "gpu_node_names": [
        ],
        "gpu_node_memories": [
        ],
        "function_names": functions,
        "function_memories": [
            250, 250, 250, 250, 250
        ],
        "function_max_delays": [
            50, 50, 50, 50, 50
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
            {"name": "f1", "users": 0, "nrt": 10},
            {"name": "f2", "users": 2, "nrt": 10},
            {"name": "f3", "users": 2, "nrt": 10},
            {"name": "f4", "users": 2, "nrt": 15},
            {"name": "f5", "users": 2, "nrt": 15}
        ],
        "edges": [
            {"source": "f1", "target": "f2", "sync": 1, "times": 1},
            {"source": "f2", "target": "f3", "sync": 2, "times": 2},
            {"source": "f2", "target": "f4", "sync": 2, "times": 1},
            {"source": "f2", "target": "f5", "sync": 3, "times": 1}
        ],
        "m": [[0, 1, 0, 0, 0], [0, 0, 2, 1, 1], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]],
        "nodes_heu": [
            {"name": "f1", "users": 0, "nrt": 10},
            {"name": "f2", "users": 2, "nrt": 10},
            {"name": "f3", "users": 2, "nrt": 10},
            {"name": "f4", "users": 2, "nrt": 15},
            {"name": "f5", "users": 2, "nrt": 15}
        ],
        "edges_heu": [
            {"source": "f1", "target": "f2", "sync": 1, "times": 1},
            {"source": "f2", "target": "f3", "sync": 2, "times": 2},
            {"source": "f2", "target": "f4", "sync": 2, "times": 1},
            {"source": "f3", "target": "f5", "sync": 3, "times": 1},
            {"source": "f4", "target": "f5", "sync": 4, "times": 1}
        ],
        "m_heu": [[0, 1, 0, 0, 0], [0, 0, 2, 1, 0], [0, 0, 0, 0, 1], [0, 0, 0, 0, 1], [0, 0, 0, 0, 0]],
        "parallel_scheduler": [[0], [1], [2, 3], [4]]
    }
