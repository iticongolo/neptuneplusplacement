from core import nodes, network

class JsonfileSockshop:

    qty_f = 7

    cpu_allocation = {}
    position = 0

    functions = ["ns/f0(Ord)", "ns/f1(Use)", "ns/f2(Cat)", "ns/f3(Shi)", "ns/f4(Pay)", "ns/f5(Uti)", "ns/f6(Del)"]
    # nodes = ["node_a", "node_b", "node_c", "node_d", "node_e", "node_f", "node_g", "node_h", "node_i", "node_j"]

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
            385, 30, 72, 21, 14, 79, 185
        ],
        "node_cores": network[position]["node_cores"],
        "gpu_node_names": [
        ],
        "gpu_node_memories": [
        ],
        "function_names": functions,
        "function_memories": [
            400, 15, 15, 350, 15, 360, 360
        ],
        "function_cold_starts": [
            3465, 270, 648, 189, 126, 711, 1665
        ],
        "function_max_delays": [
            5, 5, 5, 5, 5, 5, 5
        ],
        "gpu_function_names": [
        ],
        "gpu_function_memories": [
        ],
        "actual_cpu_allocations": cpu_allocation,
        "app": "sockshop",
        "actual_gpu_allocations": {
        },

        "nodes": [
            {"name": "f0(Ord)", "users": 0, "nrt": 10},
            {"name": "f1(Use)", "users": 1, "nrt": 15},
            {"name": "f2(Cat)", "users": 1, "nrt": 10},
            {"name": "f3(Shi)", "users": 1, "nrt": 10},
            {"name": "f4(Pay)", "users": 1, "nrt": 10},
            {"name": "f5(Uti)", "users": 1, "nrt": 10},
            {"name": "f6(Del)", "users": 1, "nrt": 10},

        ],
        "edges": [
            {"source": "f0(Ord)", "target": "f1(Use)", "sync": 1, "times": 1},
            {"source": "f0(Ord)", "target": "f2(Cat)", "sync": 1, "times": 1},
            {"source": "f0(Ord)", "target": "f3(Shi)", "sync": 1, "times": 1},
            {"source": "f0(Ord)", "target": "f4(Pay)", "sync": 1, "times": 1},
            {"source": "f0(Ord)", "target": "f5(Uti)", "sync": 2, "times": 1},
            {"source": "f0(Ord)", "target": "f6(Del)", "sync": 3, "times": 1},
        ],
        "m": [[0, 1, 1, 1, 1, 1, 1],
              [0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0]],

        "parallel_scheduler": [[0], [1, 2, 3, 4], [5], [6]]
    }
