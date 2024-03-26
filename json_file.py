import numpy as np


class Jsonfile:
    qty_f = 3
    # [[0, 20, 40],
    #  [20, 0, 10],
    #  [40, 10, 0]]

    # [[100, 0, 0], [0, 50, 0], [0, 0, 0]]
    # memory [
    #             70, 420, 22000
    #         ]

    input = {
        "with_db": False,
        "solver": {
            "type": "NeptuneMinDelay",
            "args": {"alpha": 0.0, "verbose": True}
        },
        "cpu_coeff": 1,
        "community": "community-test",
        "namespace": "namespace-test",
        "node_names": [
            "node_a", "node_b", "node_c", "node_d", "node_e", "node_f","node_g", "node_h", "node_i", "node_j"
        ],
        "node_delay_matrix": [[0., 5., 2., 5., 1., 1., 1., 1., 3., 3.],
                              [5., 0., 3., 1., 5., 4., 2., 1., 3., 3.],
                              [2., 3., 0., 2., 1., 1., 4., 3., 2., 2.],
                              [5., 1., 2., 0., 4., 2., 2., 3., 1., 2.],
                              [1., 5., 1., 4., 0., 4., 3., 4., 3., 2.],
                              [1., 4., 1., 2., 4., 0., 2., 4., 3., 2.],
                              [1., 2., 4., 2., 3., 2., 0., 3., 3., 1.],
                              [1., 1., 3., 3., 4., 4., 3., 0., 1., 4.],
                              [3., 3., 2., 1., 3., 3., 3., 1., 0., 5.],
                              [3., 3., 2., 2., 2., 2., 1., 4., 5., 0.]],
        # [[0, 20, 40],
        #  [20, 0, 10],
        #  [40, 10, 0]],
        "workload_on_source_matrix": [[100, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
        #     [[100, 0, 0], [0, 50, 0], [0, 0, 0]],

        "node_memories":[500, 600, 800, 5000, 80000, 1600, 1600, 1600, 8000, 3200],
        # [
        #     70, 420, 22000
        # ],
        "nrt": [
            200, 50, 20
        ],
        "node_cores": [800, 600, 800, 500, 8000, 1600, 1600, 1600, 8000, 3200],
        # [5000, 600, 800],
        "gpu_node_names": [
        ],
        "gpu_node_memories": [
        ],
        "function_names": [
            "ns/f0", "ns/f1", "ns/f2"
        ],
        "function_memories": [
            5, 5, 5
        ],
        "function_max_delays": [
            50, 50, 50
        ],
        "gpu_function_names": [
        ],
        "gpu_function_memories": [
        ],
        "actual_cpu_allocations": {
            "ns/f0": {
                "node_a": True,
                "node_b": True,
                "node_c": True,
                "node_d": True,
                "node_e": True,
                "node_f": True,
                "node_g": True,
                "node_h": True,
                "node_i": True,
                "node_j": True,
            },
            "ns/f1": {
                "node_a": True,
                "node_b": True,
                "node_c": True,
                "node_d": True,
                "node_e": True,
                "node_f": True,
                "node_g": True,
                "node_h": True,
                "node_i": True,
                "node_j": True,
            },
            "ns/f2": {
                "node_a": True,
                "node_b": True,
                "node_c": True,
                "node_d": True,
                "node_e": True,
                "node_f": True,
                "node_g": True,
                "node_h": True,
                "node_i": True,
                "node_j": True,
            }
        },
        "actual_gpu_allocations": {
        },
        "nodes": [
            {"name": "f0", "users": 0, "nrt": 10},
            {"name": "f1", "users": 2, "nrt": 15},
            {"name": "f2", "users": 2, "nrt": 100},
            # {"name": "f3", "users": 2, "nrt": 15},
            # {"name": "f4", "users": 2, "nrt": 15},
            # {"name": "f5", "users": 2, "nrt": 15},
            # ... additional nodes
        ],
        "edges": [
            {"source": "f0", "target": "f1", "sync": 1, "times": 2},
            {"source": "f0", "target": "f2", "sync": 2, "times": 4},
            # {"source": "f0", "target": "f3", "sync": 2, "times": 2},
            # {"source": "f0", "target": "f4", "sync": 3, "times": 2},
            # {"source": "f4", "target": "f5", "sync": 3, "times": 2},

            # ... additional edges
        ],
        "m": [[0, 1, 1], [0, 0, 0]]
    }
    #
    # input = {
    #     "with_db": False,
    #     "solver": {
    #         "type": "NeptuneMinDelay",
    #         "args": {"alpha": 0.0, "verbose": True}
    #     },
    #     "cpu_coeff": 1,
    #     "community": "community-test",
    #     "namespace": "namespace-test",
    #     "node_names": [
    #         "node_a", "node_b", "node_c"
    #     ],
    #     "node_delay_matrix": [[0, 3, 3],
    #                           [3, 0, 3],
    #                           [3, 3, 0]],
    #     "workload_on_source_matrix": [[100, 0, 100], [0, 0, 0]],
    #     "node_memories": [
    #         9, 9, 9
    #     ],
    #     "node_cores": [
    #         80, 50, 400
    #     ],
    #     "gpu_node_names": [
    #     ],
    #     "gpu_node_memories": [
    #     ],
    #     "function_names": [
    #         "ns/fn_1", "ns/fn_2"
    #     ],
    #     "function_memories": [
    #         5, 5
    #     ],
    #     "function_max_delays": [
    #         1000, 1000
    #     ],
    #     "gpu_function_names": [
    #     ],
    #     "gpu_function_memories": [
    #     ],
    #     "actual_cpu_allocations": {
    #         "ns/fn_1": {
    #             "node_a": True,
    #             "node_b": True,
    #             "node_c": True,
    #         },
    #         "ns/fn_2": {
    #             "node_a": True,
    #             "node_b": True,
    #             "node_c": True,
    #         }
    #     },
    #     "actual_gpu_allocations": {
    #     },
    # }