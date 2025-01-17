import numpy as np
import json
from collections import defaultdict

# def output_x_and_c(data, x, c):
#     x_matrix = np.empty(shape=(len(data.sources), len(data.functions), len(data.nodes)))
#     for j in range(len(data.nodes)):
#         for i in range(len(data.sources)):
#             for f in range(len(data.functions)):
#                 x_matrix[i][f][j] = x[i, f, j].solution_value()
#     c_matrix = np.empty(shape=(len(data.functions), len(data.nodes)))
#     for j in range(len(data.nodes)):
#         for f in range(len(data.functions)):
#             c_matrix[f][j] = c[f, j].solution_value()
#     return x_matrix, c_matrix

def output_x_and_c(data, x, c):
    x_matrix = np.empty(shape=(len(data.functions),len(data.sources),  len(data.nodes)))
    for f in range(len(data.functions)):
        for j in range(len(data.nodes)):
            for i in range(len(data.sources)):
                x_matrix[f][i][j] = x[f, i, j].solution_value()
    c_matrix = np.empty(shape=(len(data.functions), len(data.nodes)))
    for f in range(len(data.functions)):
        for j in range(len(data.nodes)):
            c_matrix[f][j] = c[f, j].solution_value()
    return x_matrix, c_matrix

def output_n(data, n):
    n_matrix = np.empty(shape=(len(data.nodes),))
    for j in range(len(data.nodes)):
        n_matrix[j] = n[j].solution_value()
    return n_matrix

# def convert_x_matrix(matrix, sources, functions, nodes):
#     routings = defaultdict(lambda : defaultdict(lambda : defaultdict(float)))
#     assert matrix.shape == (len(sources), len(functions), len(nodes)), f"X matrix shape malformed. matrix shape is {matrix.shape} but it should be {(len(sources), len(functions), len(nodes))}"
#     for i, source in enumerate(sources):
#         for f, function in enumerate(functions):
#             for j, destination in enumerate(nodes):
#                 if matrix[i][f][j] > 0.001:
#                     routings[source][function][destination] = np.round(matrix[i][f][j], 3)
#     return json.loads(json.dumps(routings))

def convert_x_matrix(matrix, sources, functions, nodes):
    routings = defaultdict(lambda : defaultdict(lambda : defaultdict(float)))
    assert matrix.shape == (len(functions),len(sources),  len(nodes)), f"X matrix shape malformed. matrix shape is {matrix.shape} but it should be {(len(sources), len(functions), len(nodes))}"
    for f, function in enumerate(functions):
        for i, source in enumerate(sources):
            for j, destination in enumerate(nodes):
                if matrix[f][i][j] > 0.001:
                    routings[function][source][destination] = np.round(matrix[f][i][j], 3)
    return json.loads(json.dumps(routings))


def convert_c_matrix(matrix, functions, nodes):
    allocations = defaultdict(lambda : defaultdict(bool))
    assert matrix.shape == (len(functions), len(nodes)), f"X matrix shape malformed. matrix shape is {matrix.shape} but it should be {(len(functions), len(nodes))}"
    for f, function in enumerate(functions):
        for j, destination in enumerate(nodes):
            if matrix[f][j] > 0.001:
                allocations[function][destination] = True
    return json.loads(json.dumps(allocations))