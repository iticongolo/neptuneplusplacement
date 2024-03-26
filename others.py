# import math
# import random
# import numpy as np
# # import json
# #
# # from reportlab.lib.pagesizes import letter
# # from reportlab.platypus import SimpleDocTemplate, Table, TableStyle
# # import numpy as np
# #
# # nodes = 20
# # delays = np.zeros((nodes, nodes))
# # # Define the sequence of numbers
# # sequence = [0.5, 0.8, 1.6, 3.2, 6.4, 8.0]
# #
# # # Generate random integers between 0 and 5 (corresponding to the length of the sequence minus 1)
# # random_integers = np.random.randint(0, 6, 10)
# #
# # # Map the integers to the corresponding values in the sequence
# # random_numbers = np.array([sequence[i] for i in random_integers])
# #
# # print("Random numbers following the sequence:")
# # print(random_numbers*1000)
# # array_string = "500. 600. 800. 500. 8000. 1600. 1600. 1600. 8000. 3200."
# # my_array = np.fromstring(array_string, sep=' ')
# #
# # cores_string = "500. 600. 800. 500. 8000. 1600. 1600. 1600. 8000. 3200."
# # cores = np.fromstring(cores_string, sep=' ')
# # print(cores)
# # cores = 40
# #
# # delays = np.ones((cores, cores))
# # for i in range(len(delays[0])):
# #     delays[i, i] = 0
# #     for j in range(len(delays[0])):
# #         if i != j:
# #             delays[i, j] = random.randint(1, 5)
# # aux = []
# # for i in range(len(delays[0])):
# #     for j in aux:
# #         delays[i, j] = delays[j, i]
# #     aux.append(i)
# #
# # delays = str(delays).replace(' ', ', ')
# # print(delays)
# #
# # workload = np.zeros((3, cores))
# # workload[0, 0] = 100
# #
# # print(f'workload={workload}')
# #
# # nodes = ""
# # for i in range(cores):
# #     nodes = f'{nodes}", "node{i}'
# # print(nodes)
# # # Define the dimensions of the matrix
# # rows = 3
# # cols = 4
# #
# # # Initialize an empty matrix
# # matrix = []
# #
# # # Create the matrix using a for loop
# # for i in range(rows):
# #     row = []
# #     for j in range(cols):
# #         # Append elements to the current row
# #         row.append(i * cols + j + 1)
# #     # Append the current row to the matrix
# #     matrix.append(row)
# #
# # # Convert the matrix to JSON format
# # # json_matrix = json.dumps(matrix)
# # # json_matrix[0][0]=1
# # # print(json_matrix)
# #
# #
# # # Define the list of function names
# #
# # # Custom sorting function
# # import re
# #
# # # Define the list of function names
# # function_names = ["f0", "f1", "f2", "f10", "f12", "f13", "f100", "f4text"]
# #
# # # Custom sorting function
# # def custom_sort(name):
# #     # Extract the numerical part using regular expression
# #     num_part = re.search(r'\d+', name).group()
# #     # Convert numerical part to integer
# #     num = int(num_part)
# #     return num
# #
# # # Sort the list using the custom sorting function
# # sorted_function_names = sorted(function_names, key=custom_sort)
# #
# # print(sorted_function_names)
# # cores = 45
# # delays = np.ones((cores, cores))
# #
# # # Populate delays with random values
# # for i in range(len(delays[0])):
# #     delays[i, i] = 0
# #     for j in range(len(delays[0])):
# #         if i != j:
# #             delays[i, j] = random.randint(1, 5)
# #
# # # Ensure symmetry by copying values from the upper triangle to the lower triangle
# # aux = []
# # for i in range(len(delays[0])):
# #     for j in aux:
# #         delays[i, j] = delays[j, i]
# #     aux.append(i)
# #
# # # Print each row with commas and brackets
# # for row in delays:
# #     print('[', end='')
# #     print(', '.join(map(str, row)), end='')
# #     print(']')
#
#
# def is_uniform(matrix, perc_list, avg, ones, fives, accurance):
#     one =0
#     five=0
#     valid=True
#     result=False
#     cols= len(matrix[0])
#     for col in range (cols):
#         if matrix[0,col]==1:
#             one=one+1
#         if matrix[0, col]==5:
#             five=five+1
#     for value in perc_list:
#         if abs(value-avg) >= accurance:
#             valid = False
#             break
#     if one >= ones and five >= fives and valid:
#         result = True
#     # print(f'one={one}, fives={five}, perc_list={perc_list}')
#     return result
#
# # nodes = 50
# # valid = False
# # def are_unifiform_nodes(nodes_cpu, avg, accurance):
# #     return abs(np.mean(nodes_cpu)-avg) < accurance
# # # ist of numbers to choose from
# # numbers_list = [1, 2, 4, 8]
# # avg=4
# # accurance = 0.001
# # random_numbers = []
# # while not valid:
# #     # Generate a list of size 10 with random numbers from the numbers_list
# #     random_numbers = [random.choice(numbers_list) for _ in range(nodes)]
# #     random_numbers[0] = 1
# #     valid = are_unifiform_nodes(random_numbers, avg, accurance)
# #
# #
# #
# # memory=[]
# # cpu=[]
# # print(np.mean(random_numbers))
# # for i in random_numbers:
# #     memory.append(i*2048)
# #     cpu.append(i*1000)
# # print(memory)
# # print(cpu)
#
# # a = [1000, 8000, 4000, 4000, 4000, 4000, 8000, 2000, 8000, 8000]
# # print(np.mean(a))
# # Given average value
# given_avg = 2.2
# ones=11
# fives=8
# # Define the number of cores
# cores = 50
# accurance = 0.18
# while cores < 51:
#     #Initialize delays matrix
#     delays = np.ones((cores, cores))
#     uniform = False
#     # Populate delays with random values
#     while not uniform:
#         for i in range(len(delays[0])):
#             delays[i, i] = 0
#             for j in range(len(delays[0])):
#                 if i != j:
#                     delays[i, j] = random.randint(1, 5)
#
#         # Ensure symmetry by copying values from the upper triangle to the lower triangle
#         aux = []
#         for i in range(len(delays[0])):
#             for j in aux:
#                 delays[i, j] = delays[j, i]
#             aux.append(i)
#
#         # Calculate the average of each row
#         row_avg = np.mean(delays, axis=1)
#
#         # Adjust the values in each row to match the given average
#         for i in range(len(delays)):
#             row_sum = np.sum(delays[i])
#             row_factor = given_avg / row_avg[i]
#             delays[i] *= row_factor
#             # Round the values and clip them to be within the range [1, 5]
#             delays[i] = np.clip(np.round(delays[i]), 1, 5)
#
#         # Set the diagonal values to zero
#         np.fill_diagonal(delays, 0)
#
#         aux = []
#         for i in range(len(delays[0])):
#             for j in aux:
#                 delays[i, j] = delays[j, i]
#             aux.append(i)
#         avgs = []
#         for i in range(len(delays[0])):
#             avgs.append(np.mean(delays[i]))
#
#         uniform=is_uniform(delays,avgs,given_avg,ones, fives, accurance)
#
#
#     # Print the updated delays matrix
#     # print(delays)
#     print(f'====================== {cores}=================================')
#     # Print each row with commas and brackets
#     for row in delays:
#         print('[', end='')
#         print(', '.join(map(str, row)), end='')
#         print(']')
#
#     ondes = ones + 1
#     fives = fives + 1
#     cores = cores + 5
from time import time

import matplotlib.pyplot as plt
# a=[]
# Define the size of the PDF sheet
# pdf_sheet_width = 8.5  # Width in inches
# pdf_sheet_height = 11.0  # Height in inches
#
# # Set the figure size to match the PDF sheet dimensions
# plt.figure(figsize=(pdf_sheet_width, pdf_sheet_height))
#
# # Create your plot
# plt.plot([10000, 20000, 30000], [40000, 50000, 600000])
# plt.xlabel('X-axis Label', fontsize=12)  # Adjust font size as needed
# plt.ylabel('Y-axis Label', fontsize=12)  # Adjust font size as needed
# plt.title('Your Plot Title', fontsize=14)  # Adjust font size as needed
#
# # Show or save your plot
# plt.show()  # Display the plot
# for i in range(25):
#     a.append(512)
#
# print(a)
# statistic_data = {
#         'Metric': ['Delay', 'Nodes', 'Memory'],
#         'NEP': [10, 3, 240],
#         'HEU': [9, 11, 190],
#         'ASECO': [200, 209, 400]
#     }
#     # Create a DataFrame from the data
#     df = pd.DataFrame(statistic_data)
#     # Set the 'Metric' column as index
#     df.set_index('Metric', inplace=True)
#     # Create a table plot
#     plt.figure(figsize=(8, 6))
#     table = plt.table(cellText=df.values, colLabels=df.columns, rowLabels=df.index, loc='center', cellLoc='center')
#     # Styling the table
#     table.auto_set_font_size(False)
#     table.set_fontsize(12)
#     table.scale(1.5, 1.5)
#     # Remove axis
#     plt.axis('off')
#     # Set the directory to save the plot
#     plot_directory = 'plots'
#     # Check if the directory exists, if not, create it
#     if not os.path.exists(plot_directory):
#         os.makedirs(plot_directory)
#     # Save the table as a PDF in the specified directory
#     plot_file_path = os.path.join(plot_directory, 'metric_table.pdf')
#     plt.savefig(plot_file_path, bbox_inches='tight', pad_inches=0.05)

# import matplotlib.pyplot as plt
#
# # Sample data
# x = [1, 2, 3, 4, 5]
# y = [2, 3, 5, 7, 11]
#
# # Plot with different marker shapes
# plt.plot(x, y, label='Circle', color='blue', marker='o', linestyle='-')  # Circle
# plt.plot(x, [i+1 for i in y], label='Square', color='green', marker='s', linestyle='-')  # Square
# plt.plot(x, [i+2 for i in y], label='Triangle', color='red', marker='^', linestyle='-')   # Triangle
# plt.plot(x, [i+3 for i in y], label='Star', color='orange', marker='*', linestyle='-')  # Star
# plt.plot(x, [i+4 for i in y], label='Cross', color='purple', marker='x', linestyle='-')  # Cross
#
# # Customize the plot
# plt.xlabel('X Label')
# plt.ylabel('Y Label')
# plt.title('Plot with Different Marker Shapes')
# plt.legend()
#
# # Show the plot
# plt.grid(True)
# plt.show()
#
# plt.clf()
#
# # Sample data
# x = [1, 2, 3, 4, 5]
# y = [2, 3, 5, 7, 11]
#
# # Plot with different marker shapes
# plt.plot(x, y, label='Circle', color='blue', marker='o', linestyle='-')  # Circle
# plt.plot(x, [i+1 for i in y], label='Square', color='green', marker='s', linestyle='-')  # Square
# plt.plot(x, [i+2 for i in y], label='Triangle', color='red', marker='^', linestyle='-')   # Triangle
# plt.plot(x, [i+3 for i in y], label='Star', color='orange', marker='*', linestyle='-')  # Star
# plt.plot(x, [i+4 for i in y], label='Cross', color='purple', marker='x', linestyle='-')  # Cross
# plt.plot(x, [i+5 for i in y], label='Diamond', color='cyan', marker='D', linestyle='-')  # Diamond
#
# # Customize the plot
# plt.xlabel('X Label')
# plt.ylabel('Y Label')
# plt.title('Plot with Different Marker Shapes')
# plt.legend()
#
# # Show the plot
# plt.grid(True)
# plt.show()
#


import heapq
import itertools

import itertools

import itertools

import itertools

# import itertools
#
# def total_distance(graph, path):
#     distance = 0
#     for i in range(len(path) - 1):
#         distance += graph[path[i]][path[i+1]]
#     return distance
#
# def held_karp(graph):
#     n = len(graph)
#     nodes = list(range(n))
#     shortest_distance = float('inf')
#     shortest_path = []
#
#     for perm in itertools.permutations(nodes):
#         if perm[0] == 0:  # Ensure the start node is always at the beginning
#             distance = total_distance(graph, perm)
#             if distance < shortest_distance:
#                 shortest_distance = distance
#                 shortest_path = perm
#
#     return shortest_distance, shortest_path
#
# # Example usage:
# topology = [
#     [0, 13, 9, 10],
#     [2, 0, 6, 14],
#     [9, 6, 0, 15],
#     [10, 4, 5, 0]
# ]
#
# # Calculate shortest path
# shortest_distance, shortest_path = held_karp(topology)
#
# # Convert node indices to node labels (e.g., 0 to 'a', 1 to 'b', etc.)
# node_labels = [chr(ord('a') + node) for node in shortest_path]
#
# # Print the shortest path
# print("Shortest path:", " -> ".join(node_labels))
# print("Shortest distance:", shortest_distance)

# import itertools
#
# def total_distance(graph, path):
#     distance = 0
#     for i in range(len(path) - 1):
#         distance += graph[path[i]][path[i+1]]
#     return distance
#
# def held_karp_with_exclusions(graph, start, destination_nodes, excluded_nodes):
#     n = len(graph)
#     nodes = [node for node in destination_nodes if node not in excluded_nodes]
#     shortest_distance = float('inf')
#     shortest_path = []
#
#     for perm in itertools.permutations(nodes):
#         perm = (start,) + perm  # Ensure the start node is always at the beginning
#         distance = total_distance(graph, perm)
#         if distance < shortest_distance:
#             shortest_distance = distance
#             shortest_path = perm
#
#     return shortest_distance, shortest_path
#
# # Example usage:
# # topology = [
# #         [0, 5, 1, 1, 2, 1, 3, 4, 1, 1],
# #         [5, 0, 1, 3, 1, 5, 1, 2, 1, 1],
# #         [1, 1, 0, 5, 1, 3, 1, 4, 1, 1],
# #         [1, 3, 5, 0, 1, 1, 3, 1, 2, 1],
# #         [2, 1, 1, 1, 0, 1, 2, 1, 1, 5],
# #         [1, 5, 3, 1, 1, 0, 1, 3, 1, 1],
# #         [3, 1, 1, 3, 2, 1, 0, 1, 4, 1],
# #         [4, 2, 4, 1, 1, 3, 1, 0, 1, 1],
# #         [1, 1, 1, 2, 1, 1, 4, 1, 0, 2],
# #         [1, 1, 1, 1, 5, 1, 1, 1, 2, 0]]
#
# topology = [
#         [0, 10, 5, 3, 8, 7],
#         [6, 0, 1, 1, 2, 8],
#         [10, 20, 0, 10, 7, 3],
#         [10, 3, 8, 0, 1, 1],
#         [20, 10, 1, 1, 0, 8],
#         [7, 7, 8, 9, 7, 0]]
#
# start_node = 0  # Start from node 'a'
# destination_nodes = [1, 2, 3, 4, 5]  # Example destination nodes: b to d
# excluded_nodes = [1]  # Exclude node 'b'
# time1 = time()
# shortest_distance, shortest_path = held_karp_with_exclusions(topology, start_node, destination_nodes, excluded_nodes)
# endtime = time()
# print(f'Time={endtime - time1}')
#
# # Print the shortest path with node indices
# print("Shortest path:", shortest_path)
# print("Shortest distance:", shortest_distance)
#


# import itertools
#
# def total_distance(graph, path):
#     distance = 0
#     for i in range(len(path) - 1):
#         distance += graph[path[i]][path[i+1]]
#     return distance
#
# def held_karp_with_exclusions(graph, start, destination_nodes, excluded_nodes):
#     n = len(graph)
#     nodes = [node for node in destination_nodes if node not in excluded_nodes]
#     all_paths = []
#
#     for perm in itertools.permutations(nodes):
#         perm = (start,) + perm  # Ensure the start node is always at the beginning
#         distance = total_distance(graph, perm)
#         all_paths.append((distance, perm))
#
#     all_paths.sort()
#     return all_paths
#
# # Example usage:
# topology = [
#     [0, 10, 5, 3, 8, 7],
#     [6, 0, 1, 1, 2, 8],
#     [10, 20, 0, 10, 7, 3],
#     [10, 3, 8, 0, 1, 1],
#     [20, 10, 1, 1, 0, 8],
#     [7, 7, 8, 9, 7, 0]]
#
# start_node = 1  # Start from node 'a'
# destination_nodes = [1, 3, 4, 5]  # Example destination nodes: b to d
# excluded_nodes = [3, 1]  # Exclude node 'b'
#
# # Calculate paths sorted by length
# all_paths = held_karp_with_exclusions(topology, start_node, destination_nodes, excluded_nodes)
#
# # Print all paths from shortest to longest
# for distance, path in all_paths:
#     print("Path:", path, "Distance:", distance)

#
# import itertools
# from time import time
#
# def total_distance(graph, path):
#     distance = 0
#     for i in range(len(path) - 1):
#         distance += graph[path[i]][path[i+1]]
#     return distance
#
# def held_karp_with_exclusions(graph, start, destination_nodes, excluded_nodes):
#     n = len(graph)
#     nodes = [node for node in destination_nodes if node not in excluded_nodes]
#     shortest_distance = float('inf')
#     shortest_path = []
#
#     for perm in itertools.permutations(nodes):
#         perm = (start,) + perm  # Ensure the start node is always at the beginning
#         distance = total_distance(graph, perm)
#         if distance < shortest_distance:
#             shortest_distance = distance
#             shortest_path = perm
#
#     return shortest_distance, shortest_path
#
# # Example usage:
# topology = [
#     [0, 1, 3, 1, 1, 1, 2, 2, 4, 2, 5, 2, 1, 1, 5, 5, 1, 2, 2, 4],
#     [1, 0, 4, 2, 1, 3, 3, 4, 4, 1, 2, 1, 1, 4, 3, 1, 1, 4, 3, 3],
#     [3, 4, 0, 3, 3, 3, 3, 1, 2, 1, 2, 3, 3, 1, 1, 3, 3, 2, 2, 2],
#     [1, 2, 3, 0, 3, 2, 2, 2, 3, 1, 1, 3, 2, 4, 4, 2, 3, 2, 3, 1],
#     [1, 1, 3, 3, 0, 2, 4, 2, 2, 2, 1, 4, 2, 2, 2, 3, 2, 3, 1, 3],
#     [1, 3, 3, 2, 2, 0, 1, 1, 1, 5, 4, 2, 2, 2, 1, 3, 4, 1, 3, 5],
#     [2, 3, 3, 2, 4, 1, 0, 3, 1, 4, 1, 3, 3, 1, 1, 3, 1, 3, 3, 2],
#     [2, 4, 1, 2, 2, 1, 3, 0, 1, 3, 1, 2, 4, 3, 3, 3, 2, 3, 1, 3],
#     [4, 4, 2, 3, 2, 1, 1, 1, 0, 2, 4, 2, 2, 2, 2, 1, 4, 2, 3, 3],
#     [2, 1, 1, 1, 2, 5, 4, 3, 2, 0, 2, 4, 3, 2, 4, 2, 1, 2, 1, 1],
#     [5, 2, 2, 1, 1, 4, 1, 1, 4, 2, 0, 3, 1, 2, 1, 5, 4, 2, 2, 2],
#     [2, 1, 3, 3, 4, 2, 3, 2, 2, 4, 3, 0, 1, 3, 1, 2, 2, 2, 4, 1],
#     [1, 1, 3, 2, 2, 2, 3, 4, 2, 3, 1, 1, 0, 3, 4, 3, 1, 2, 1, 3],
#     [1, 4, 1, 4, 2, 2, 1, 3, 2, 2, 2, 3, 3, 0, 2, 2, 2, 2, 4, 2],
#     [5, 3, 1, 4, 2, 1, 1, 3, 2, 4, 1, 1, 4, 2, 0, 2, 4, 4, 2, 1],
#     [5, 1, 3, 2, 3, 3, 3, 3, 1, 2, 5, 2, 3, 2, 2, 0, 3, 1, 1, 1],
#     [1, 1, 3, 3, 2, 4, 1, 2, 4, 1, 4, 2, 1, 2, 4, 3, 0, 3, 3, 2],
#     [2, 4, 2, 2, 3, 1, 3, 3, 2, 2, 2, 2, 2, 2, 4, 1, 3, 0, 2, 4],
#     [2, 3, 2, 3, 1, 3, 3, 1, 3, 1, 2, 4, 1, 4, 2, 1, 3, 2, 0, 3],
#     [4, 3, 2, 1, 3, 5, 2, 3, 3, 1, 2, 1, 3, 2, 1, 1, 2, 4, 3, 0]]
#
# start_node = 0  # Start from node 'a'
# destination_nodes = [node+1 for node in range(len(topology[0])-10)]  # Example destination nodes: b to d
# excluded_nodes = [0]  # Exclude node 'b'
#
# # Calculate shortest path
# time1 = time()
# shortest_distance, shortest_path = held_karp_with_exclusions(topology, start_node, destination_nodes, excluded_nodes)
# endtime = time()
# print(f'Time={endtime - time1}')
#
# # Print the shortest path with node indices
# print(f'Shortest path: {len(shortest_path)}')
# print("Shortest distance:", shortest_path)

print(3*False)