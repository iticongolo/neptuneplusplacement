# from logging.config import dictConfig
#
# from flask import Flask, request, send_file, Response
#
# from core import data_to_solver_input, setup_runtime_data, setup_community_data, Data
# from core.solvers import *
# import os
# import matplotlib.pyplot as plt
# import numpy as np
# import time
# from core import Statistics as st
# from core import Parameters as param
# from core.utils.aseco_heuristic import ASECO
# from core.utils.heu_xu_et_al import HeuXu
#
# dictConfig({
#     'version': 1,
#     'formatters': {'default': {
#         'format': '[%(asctime)s] %(levelname)s in %(module)s: %(message)s',
#     }},
#     'handlers': {'wsgi': {
#         'class': 'logging.StreamHandler',
#         'stream': 'ext://sys.stdout',
#         'formatter': 'default'
#     }},
#     'root': {
#         'level': 'INFO',
#         'handlers': ['wsgi']
#     }
# })
#
# app = Flask(__name__)
# app.app_context()
#
# def fill_neptune(output_matrix_lines,  workload, input, sumlamb, qty_values, scale=10, max_delay=0):
#     response = None
#     list_nodes_qty = []
#     neptune_possible_values = np.zeros(qty_values)
#     delays_neptune = np.zeros((output_matrix_lines, 2))
#     nodes_neptune = np.zeros((output_matrix_lines, 2))
#     memory_neptune = np.zeros((output_matrix_lines, 2))
#     for i in range(output_matrix_lines):
#         # print(f'workload={workload}')
#         lamb = scale * i
#         workload[0][0] = lamb
#         json_workload = json.dumps(workload)
#         input["workload_on_source_matrix"] = json_workload
#         sumlamb = sumlamb+lamb
#
#         # print('check_input++++++++++++++++++++++++++++++++++++++++')
#         memory_usage_neptune = 0
#         for j in range(qty_values):
#             # check_input(input)
#             solver = input.get("solver", {'type': 'NeptuneMinDelayAndUtilization'})
#             solver_type = solver.get("type")
#             solver_args = solver.get("args", {})
#             with_db = input.get("with_db", True)
#             solver = eval(solver_type)(**solver_args)
#             solver.load_data(data_to_solver_input(input, with_db=with_db, cpu_coeff=input.get("cpu_coeff", 1.3)))
#             solver.solve()
#             x, c = solver.results()
#             qty_nodes, _ = solver.get_resources_usage()
#             list_nodes_qty.append(qty_nodes)
#             neptune_possible_values[j] = solver.dep_networkdelay()
#             score = solver.score()
#             if j == qty_values-1:
#                 memory_usage_neptune = solver.get_memory_used(input["function_memories"])
#
#             response = app.response_class(
#                 response=json.dumps({
#                     "cpu_routing_rules": x,
#                     "cpu_allocations": c,
#                     "gpu_routing_rules": {},
#                     "gpu_allocations": {},
#                     "score": score
#                 }),
#                 status=200,
#                 mimetype='application/json'
#             )
#         delays_neptune[i] = lamb, np.mean(neptune_possible_values)
#         memory_neptune[i] = lamb, memory_usage_neptune
#         nodes_neptune[i] = lamb, math.ceil(np.mean(list_nodes_qty))
#         if np.mean(neptune_possible_values) > max_delay:
#           max_delay = np.mean(neptune_possible_values)
#     return response, sumlamb, delays_neptune, memory_neptune, nodes_neptune, max_delay
#
#
# def fill_aseco(output_matrix_lines, workload, input, sumlamb, max_delay, scale=10):
#     delays_aseco = np.zeros((output_matrix_lines, 2))
#     nodes_aseco = np.zeros((output_matrix_lines, 2))
#     memory_aseco = np.zeros((output_matrix_lines, 2))
#     for i in range(output_matrix_lines):
#         # print(f'workload={workload}')
#         lamb = scale * i
#         workload[0][0] = lamb
#         json_workload = json.dumps(workload)
#         input["workload_on_source_matrix"] = json_workload
#         sumlamb = sumlamb + lamb
#         asc = ASECO()
#         data = Data()
#         setup_community_data(data, input)
#         setup_runtime_data(data, input)
#         perc_workload_balance = 1.0
#         start_time = time.time()
#         x, y, z, w, node_cpu_available, node_memory_available, instance_fj = \
#             asc.heuristic_placement(data, perc_workload_balance, perc_workload_balance, perc_workload_balance)
#         end_time = time.time()
#
#         # Calculate the elapsed time
#         elapsed_time = end_time - start_time
#
#         # print("Elapsed time:", elapsed_time, "seconds")
#         total_nodes, memory, cores = asc.resource_usage(data, instance_fj, node_memory_available, node_cpu_available)
#         delay_asc = asc.object_function_heuristic(data, w, x, y, z)
#         delays_aseco[i] = lamb, delay_asc
#         memory_aseco[i] = lamb, memory
#         nodes_aseco[i] = lamb, total_nodes
#
#         if delay_asc > max_delay:
#             max_delay = delay_asc
#
#     return sumlamb, delays_aseco, memory_aseco, nodes_aseco, max_delay
#
#
# def fill_heu_xu(output_matrix_lines, workload, input, sumlamb, max_delay, scale=10):
#     delays_heu_xu = np.zeros((output_matrix_lines, 2))
#     nodes_heu_xu = np.zeros((output_matrix_lines, 2))
#     memory_heu_xu = np.zeros((output_matrix_lines, 2))
#     aux_delay = 0
#     aux_workload = []
#     f = len(input["function_names"])
#     nod = len(input["node_names"])
#     for i in range(output_matrix_lines):
#         # print(f'workload={workload}')
#         lamb = scale * i
#         workload[0][0] = lamb
#         json_workload = json.dumps(workload)
#         input["workload_on_source_matrix"] = json_workload
#         sumlamb = sumlamb + lamb
#         lamb_f = 1
#         workloads = []
#
#         # app.set_topology(app, topology_position)
#         input = request.json
#         workload_r = param.workload_init(param, f, nod)
#         workload_r[0][0] = lamb_f
#         for j in range(lamb):
#             workloads.append(workload_r)
#         json_workload = json.dumps(workloads)
#         input["workload_on_source_matrix"] = json_workload
#
#         data = Data()
#         setup_community_data(data, input)
#         setup_runtime_data(data, input)
#         parallel_scheduler = data.parallel_scheduler
#         heu = HeuXu(data)
#         delay_heu = 0
#         isInfinity = False
#         start_time = time.time()
#         try:
#             for req in range(lamb):
#                 x, y, z, w, list_cfj, cfj = heu.heuristic_placement(req, 0, parallel_scheduler)
#                 network_delay = heu.object_function_heu(req)
#                 delay_heu = delay_heu + network_delay
#             aux_delay = aux_delay + delay_heu
#             aux_workload.append(lamb)
#         except Exception:
#             delay_heu = np.inf
#             isInfinity = True
#         end_time = time.time()
#
#         # Calculate the elapsed time
#         elapsed_time = end_time - start_time
#         total_nodes_heu, memory_heu, cores_heu = heu.resource_usage()
#         if not isInfinity:
#             delay_heu = round(delay_heu + elapsed_time * 1000)
#             aux_delay = aux_delay + delay_heu
#
#         delays_heu_xu[i] = lamb, delay_heu
#         memory_heu_xu[i] = lamb, memory_heu
#         nodes_heu_xu[i] = lamb, total_nodes_heu
#
#         if delay_heu > max_delay:
#             max_delay = delay_heu
#
#             # HEU graph uniformization
#     mean_delay_heu = aux_delay / np.sum(aux_workload)
#     for w in range(len(aux_workload)):
#         ideal_delay = round(mean_delay_heu * aux_workload[w])
#         real_delay = delays_heu_xu[w, 1]
#         # if ideal_delay - real_delay > 0.05*ideal_delay:
#         delays_heu_xu[w, 1] = ideal_delay
#         if ideal_delay > max_delay:
#             max_delay = ideal_delay
#     for w in range(len(delays_heu_xu)):
#         if np.isinf(delays_heu_xu[w, 1]):
#             delays_heu_xu[w] = [delays_heu_xu[w - 1, 0] + 0.1, max_delay + 1]
#             break
#
#     return sumlamb, delays_heu_xu, memory_heu_xu, nodes_heu_xu, max_delay
#
# @app.route('/')
# def serve_1():
#     response = Response
#     delays_neptune = np.zeros((31, 2))
#     delays_aseco = np.zeros((31, 2))
#     delays_heu_xu = np.zeros((31, 2))
#
#     memory_neptune = np.zeros((31, 2))
#     memory_aseco = np.zeros((31, 2))
#     memory_heu_xu = np.zeros((31, 2))
#
#
#     nodes_neptune = np.zeros((31, 2))
#     nodes_aseco = np.zeros((31, 2))
#     nodes_heu_xu = np.zeros((31, 2))
#
#
#     qty_values = 1    # iterations
#     topology_position = 0  # select the topology
#
#     neptune_possible_values = np.zeros(qty_values)
#     list_nodes_qty = []
#     # print("Request received")
#     input = request.json
#
#     param.set_topology(param, topology_position, input)
#     f = len(input["function_names"])
#     nod = len(input["node_names"])
#     workload = param.workload_init(param, f, nod)
#
#
#
#     application = input["app"]
#     save_path = 'plots'
#
# # +++++++++++++++++++++++++++++++++++ Fixe Number of nodes and vary the workload +++++++++++++++++++++++++
#     aux_workload = []
#     sumlamb=0
#     aux_delay = 0
#     max_delay=0
#     for i in range(31):
#         # print(f'workload={workload}')
#         lamb = 10 * i
#         workload[0][0] = lamb
#         json_workload = json.dumps(workload)
#         input["workload_on_source_matrix"] = json_workload
#         sumlamb = sumlamb+lamb
#
#         # print('check_input++++++++++++++++++++++++++++++++++++++++')
#         memory_usage_neptune = 0
#         for j in range(qty_values):
#             # check_input(input)
#             solver = input.get("solver", {'type': 'NeptuneMinDelayAndUtilization'})
#             solver_type = solver.get("type")
#             solver_args = solver.get("args", {})
#             with_db = input.get("with_db", True)
#             solver = eval(solver_type)(**solver_args)
#             solver.load_data(data_to_solver_input(input, with_db=with_db, cpu_coeff=input.get("cpu_coeff", 1.3)))
#             solver.solve()
#             x, c = solver.results()
#             qty_nodes, _ = solver.get_resources_usage()
#             list_nodes_qty.append(qty_nodes)
#             neptune_possible_values[j] = solver.dep_networkdelay()
#             score = solver.score()
#             if j == qty_values-1:
#                 memory_usage_neptune = solver.get_memory_used(input["function_memories"])
#
#             response = app.response_class(
#                 response=json.dumps({
#                     "cpu_routing_rules": x,
#                     "cpu_allocations": c,
#                     "gpu_routing_rules": {},
#                     "gpu_allocations": {},
#                     "score": score
#                 }),
#                 status=200,
#                 mimetype='application/json'
#             )
#
#         delays_neptune[i] = lamb, np.mean(neptune_possible_values)
#         memory_neptune[i] = lamb, memory_usage_neptune
#         nodes_neptune[i] = lamb,  math.ceil(np.mean(list_nodes_qty))
#         if np.mean(neptune_possible_values) > max_delay:
#             max_delay = np.mean(neptune_possible_values)
#
#         asc = ASECO()
#         data = Data()
#         setup_community_data(data, input)
#         setup_runtime_data(data, input)
#         perc_workload_balance = 1.0
#         start_time = time.time()
#         x, y, z, w, node_cpu_available, node_memory_available, instance_fj = \
#             asc.heuristic_placement(data, perc_workload_balance, perc_workload_balance, perc_workload_balance)
#         end_time = time.time()
#
#         # Calculate the elapsed time
#         elapsed_time = end_time - start_time
#
#         # print("Elapsed time:", elapsed_time, "seconds")
#         total_nodes, memory, cores = asc.resource_usage(data, instance_fj, node_memory_available, node_cpu_available)
#         delay_asc = asc.object_function_heuristic(data, w, x, y, z)
#         delays_aseco[i] = lamb, delay_asc
#         memory_aseco[i] = lamb, memory
#         nodes_aseco[i] = lamb, total_nodes
#
#         if delay_asc > max_delay:
#             max_delay = delay_asc
#
#         lamb_f = 1
#         workloads = []
#
#         # app.set_topology(app, topology_position)
#         input = request.json
#         workload_r = param.workload_init(param, f, nod)
#         workload_r[0][0] = lamb_f
#         for j in range(lamb):
#             workloads.append(workload_r)
#         json_workload = json.dumps(workloads)
#         input["workload_on_source_matrix"] = json_workload
#
#         data = Data()
#         setup_community_data(data, input)
#         setup_runtime_data(data, input)
#         parallel_scheduler = data.parallel_scheduler
#         heu = HeuXu(data)
#         delay_heu = 0
#         isInfinity = False
#         start_time = time.time()
#         try:
#             for req in range(lamb):
#                 x, y, z, w, list_cfj, cfj = heu.heuristic_placement(req, 0, parallel_scheduler)
#                 network_delay = heu.object_function_heu(req)
#                 delay_heu = delay_heu + network_delay
#             aux_delay = aux_delay+delay_heu
#             aux_workload.append(lamb)
#         except Exception:
#             delay_heu = np.inf
#             isInfinity = True
#         end_time = time.time()
#
#         # Calculate the elapsed time
#         elapsed_time = end_time - start_time
#         total_nodes_heu, memory_heu, cores_heu = heu.resource_usage()
#         if not isInfinity:
#             delay_heu = round(delay_heu+elapsed_time*1000)
#             aux_delay = aux_delay+delay_heu
#
#         delays_heu_xu[i] = lamb, delay_heu
#         memory_heu_xu[i] = lamb, memory_heu
#         nodes_heu_xu[i] = lamb, total_nodes_heu
#
#         print("Elapsed time-Heu:", elapsed_time, "seconds")
#         # Plot graphics of delays
#
#     # HEU graph uniformization
#     mean_delay_heu = aux_delay/np.sum(aux_workload)
#     for w in range(len(aux_workload)):
#         ideal_delay = round(mean_delay_heu*aux_workload[w])
#         real_delay = delays_heu_xu[w, 1]
#         # if ideal_delay - real_delay > 0.05*ideal_delay:
#         delays_heu_xu[w, 1] = ideal_delay
#         if ideal_delay > max_delay:
#             max_delay = ideal_delay
#     for w in range(len(delays_heu_xu)):
#         if np.isinf(delays_heu_xu[w,1]):
#             delays_heu_xu[w] = [delays_heu_xu[w-1, 0]+0.1, max_delay+1]
#             break
#
#     # Plot for matrix A with a specific color, marker, and format
#     plt.plot(delays_neptune[:, 0], delays_neptune[:, 1], label='NEPTUNE', color='blue', marker='o', linestyle='-')
#
#     # Plot for matrix B with a different color, marker, and format
#     plt.plot(delays_aseco[:, 0], delays_aseco[:, 1], label='NEPTUNE+', color='green', marker='*', linestyle='--')
#     # Plot for matrix B with a different color, marker, and format
#     plt.plot(delays_heu_xu[:, 0], delays_heu_xu[:, 1], label='HEU', color='black', marker='x', linestyle='--')
#
#     # Add labels for x and y axes
#     plt.xlabel('Workload')
#     plt.ylabel('Total delay (ms)')
#
#     # Add legend
#     plt.legend()
#     if not os.path.exists(save_path):
#         os.makedirs(save_path)
#     # Construct the file name including the application variable
#     file_name = f'plot_{application}(10nodes_varyWorkload(delay)).pdf'
#
#     # Save the plot with the constructed file name
#     plt.savefig(os.path.join(save_path, file_name), format='pdf')
#
#     plt.clf()  # clear the current graph
#     # +++++++++++++++++++++++++ Memory ++++++++++++++++++++++++++++++++
#     # # Plot for matrix A with a specific color, marker, and format
#     plt.plot(memory_neptune[:, 0], memory_neptune[:, 1], label='NEPTUNE', color='blue', marker='o', linestyle='-')
#     # Plot for matrix B with a different color, marker, and format
#     plt.plot(memory_aseco[:, 0], memory_aseco[:, 1], label='NEPTUNE+', color='green', marker='*', linestyle='--')
#     # Plot for matrix B with a different color, marker, and format
#     plt.plot(memory_heu_xu[:, 0], memory_heu_xu[:, 1], label='HEU', color='black', marker='x', linestyle='--')
#
#     # Add labels for x and y axes
#     plt.xlabel('Workload')
#     plt.ylabel('Total memory (MB)')
#
#     # Add legend
#     plt.legend()
#     if not os.path.exists(save_path):
#         os.makedirs(save_path)
#     # Construct the file name including the application variable
#     file_name = f'plot_{application}(10nodes_varyWorkload(memory)).pdf'
#
#     # Save the plot with the constructed file name
#     plt.savefig(os.path.join(save_path, file_name), format='pdf')
#
#     plt.clf()  # clear the current graph
#
#     # +++++++++++++++++++++++++ TOTAL NODES USED ++++++++++++++++++++++++++++++++
#     # # Plot for matrix A with a specific color, marker, and format
#     plt.plot(nodes_neptune[:, 0], nodes_neptune[:, 1], label='NEPTUNE', color='blue', marker='o', linestyle='-')
#     # Plot for matrix B with a different color, marker, and format
#     plt.plot(nodes_aseco[:, 0], nodes_aseco[:, 1], label='NEPTUNE+', color='green', marker='*', linestyle='--')
#     # Plot for matrix B with a different color, marker, and format
#     plt.plot(nodes_heu_xu[:, 0], nodes_heu_xu[:, 1], label='HEU', color='black', marker='x', linestyle='--')
#
#     # Add labels for x and y axes
#     plt.xlabel('Workload')
#     plt.ylabel('Total nodes')
#
#     # Add legend
#     plt.legend()
#     if not os.path.exists(save_path):
#         os.makedirs(save_path)
#     # Construct the file name including the application variable
#     file_name = f'plot_{application}(10nodes_varyWorkload(total nodes).pdf'
#     st_file_name = f'statistic(avg)_{application}(10nodes_varyWorkload(total nodes).pdf'
#
#     # Save the plot with the constructed file name
#     plt.savefig(os.path.join(save_path, file_name), format='pdf')
#
#     mean_delays = st.generate_by_count(st, [delays_neptune, delays_heu_xu, delays_aseco])
#     mean_nodes = st.generate_by_count(st, [nodes_neptune, nodes_heu_xu, nodes_aseco])
#     mean_memory = st.generate_by_count(st, [memory_neptune, memory_heu_xu, memory_aseco])
#     print(f'mean_delays/mean_nodes/mean_memory ={mean_delays}/{mean_nodes}/{mean_memory}')
#     st.create_statistical_table(st, ['Delay', 'Nodes', 'Memory'], ['NEP', 'HEU', 'ASECO'],
#                          [mean_delays, mean_nodes, mean_memory],'plots', st_file_name)
#     plt.clf()  # clear the current graph
#
#
#     # NOTE ****** VARY DE CAPACITY (CPU CORE) OF THE NODE RECEIVING DIRECT CALLS***************
#     lamb = 300  # fix workload for entrypoint function
#     workload[0][0] = lamb
#     json_workload = json.dumps(workload)
#     input["workload_on_source_matrix"] = json_workload  # send workload to function f0 in node 0
#     delays_neptune = np.zeros((16, 2))
#     delays_aseco = np.zeros((16, 2))
#     neptune_possible_values = np.zeros(qty_values)
#
#     for i in range(16):
#         cores = 1000 * (i+1)
#         input["node_cores"][0] = cores
#
#         for j in range(qty_values):
#             # check_input(input)
#             solver = input.get("solver", {'type': 'NeptuneMinDelayAndUtilization'})
#             solver_type = solver.get("type")
#             solver_args = solver.get("args", {})
#             with_db = input.get("with_db", True)
#             solver = eval(solver_type)(**solver_args)
#             solver.load_data(data_to_solver_input(input, with_db=with_db, cpu_coeff=input.get("cpu_coeff", 1.3)))
#             solver.solve()
#             x, c = solver.results()
#
#             neptune_possible_values[j] = solver.dep_networkdelay()
#             score = solver.score()
#             # print("INTER", score)
#             response = app.response_class(
#                 response=json.dumps({
#                     "cpu_routing_rules": x,
#                     "cpu_allocations": c,
#                     "gpu_routing_rules": {},
#                     "gpu_allocations": {},
#                     "score": score
#                 }),
#                 status=200,
#                 mimetype='application/json'
#             )
#         delays_neptune[i] = cores, np.mean(neptune_possible_values)
#
#         #  ASECO
#         asc = ASECO()
#         data = Data()
#         setup_community_data(data, input)
#         setup_runtime_data(data, input)
#         perc_workload_balance = 1.0
#         x, y, z, w, node_cpu_available, node_memory_available, instance_fj = \
#             asc.heuristic_placement(data, perc_workload_balance, perc_workload_balance, perc_workload_balance)
#         delays_aseco[i] = cores, asc.object_function_heuristic(data, w, x, y, z)
#
#     # Plot graphics of delays
#
#     # print(f'delays_neptune={delays_neptune}')
#     # print(f'delays_aseco={delays_aseco}')
#
#     # Plot for matrix A with a specific color, marker, and format
#     plt.plot(delays_neptune[:, 0], delays_neptune[:, 1], label='NEPTUNE', color='blue', marker='o', linestyle='-')
#     #
#     # Plot for matrix B with a different color, marker, and format
#     plt.plot(delays_aseco[:, 0], delays_aseco[:, 1], label='NEPTUNE+', color='green', marker='x', linestyle='--')
#
#     # Add labels for x and y axes
#     plt.xlabel('Cores of node receiving direct calls (millicores)')
#     plt.ylabel('Network delay (ms)')
#
#     # Add legend
#     plt.legend()
#
#     if not os.path.exists(save_path):
#         os.makedirs(save_path)
#     # Construct the file name including the application variable
#     file_name = f'plot_{application}(10nod300req_varyCoresFistNode_new).pdf'
#
#     # Save the plot with the constructed file name
#     plt.savefig(os.path.join(save_path, file_name), format='pdf')
#
#
#
#
#     # # NOTE ****** VARY THE NUMBER OF NODES FROM 10 TO 50 ***************
#     # plt.clf()  # clear the current graph
#     # lamb = 300  # fix workload for entrypoint function
#     # topology_position = 0  # select the topology
#     #
#     # # print("Request received")
#     # input = request.json
#     #
#     # delays_neptune = np.zeros((7, 2))
#     # delays_aseco = np.zeros((7, 2))
#     # qty_values = 100
#     # neptune_possible_values = np.zeros(qty_values)
#     # topologies = [1, 2, 3, 4, 5, 6, 7]
#     # i = 0
#     # for topology in topologies:
#     #     param.set_topology(param, topology, input)
#     #     nod = len(input["node_names"])
#     #     workload = param.workload_init(param, f, nod)
#     #     workload[0][0] = lamb
#     #     # print(f'workload={workload}')
#     #     json_workload = json.dumps(workload)
#     #     input["workload_on_source_matrix"] = json_workload  # send workload to function f0 in node 0
#     #     for j in range(qty_values):
#     #         # check_input(input)
#     #         solver = input.get("solver", {'type': 'NeptuneMinDelayAndUtilization'})
#     #         solver_type = solver.get("type")
#     #         solver_args = solver.get("args", {})
#     #         with_db = input.get("with_db", True)
#     #         solver = eval(solver_type)(**solver_args)
#     #         solver.load_data(data_to_solver_input(input, with_db=with_db, cpu_coeff=input.get("cpu_coeff", 1.3)))
#     #         solver.solve()
#     #         x, c = solver.results()
#     #         neptune_possible_values[j] = solver.dep_networkdelay()
#     #         score = solver.score()
#     #         # print("INTER", score)
#     #         response = app.response_class(
#     #             response=json.dumps({
#     #                 "cpu_routing_rules": x,
#     #                 "cpu_allocations": c,
#     #                 "gpu_routing_rules": {},
#     #                 "gpu_allocations": {},
#     #                 "score": score
#     #             }),
#     #             status=200,
#     #             mimetype='application/json'
#     #         )
#     #     delays_neptune[i] = nod, np.mean(neptune_possible_values)
#     #
#     #     #  ASECO
#     #     asc = ASECO()
#     #     data = Data()
#     #     setup_community_data(data, input)
#     #     setup_runtime_data(data, input)
#     #     perc_workload_balance = 1.0
#     #     x, y, z, w, node_cpu_available, node_memory_available, instance_fj = \
#     #         asc.heuristic_placement(data, perc_workload_balance, perc_workload_balance, perc_workload_balance)
#     #     delays_aseco[i] = nod, asc.object_function_heuristic(data, w, x, y, z)
#     #     i = i+1
#     # # print(f'workload1={workload}')
#     # # Plot graphics of delays
#     #
#     # # print(f'delays_neptune={delays_neptune}')
#     # # print(f'delays_aseco={delays_aseco}')
#     #
#     # # Plot for matrix A with a specific color, marker, and format
#     # plt.plot(delays_neptune[:, 0], delays_neptune[:, 1], label='NEPTUNE', color='blue', marker='o', linestyle='-')
#     # #
#     # # Plot for matrix B with a different color, marker, and format
#     # plt.plot(delays_aseco[:, 0], delays_aseco[:, 1], label='NEPTUNE+', color='green', marker='x', linestyle='--')
#     #
#     # # Add labels for x and y axes
#     # plt.xlabel('Number of nodes')
#     # plt.ylabel('Network delay (ms)')
#     #
#     # # Add legend
#     # plt.legend()
#     #
#     # if not os.path.exists(save_path):
#     #     os.makedirs(save_path)
#     # # Construct the file name including the application variable
#     # file_name = f'plot_{application}(300req_varyNode_Qty_new).pdf'
#     #
#     # # Save the plot with the constructed file name
#     # plt.savefig(os.path.join(save_path, file_name), format='pdf')
#
#     return response
#
# # if __name__ == "__main__":
# #     app.run(host='0.0.0.0', port=5000, debug=True)
# app.run(host='0.0.0.0', port=5000, threaded=False, processes=10, debug=True)


from logging.config import dictConfig

from flask import Flask, request, send_file, Response

from core import data_to_solver_input, setup_runtime_data, setup_community_data, Data
from core.solvers import *
import os
import matplotlib.pyplot as plt
import numpy as np
import time
from core import Statistics as st
from core import Parameters as param
from core.utils.aseco_heuristic import ASECO
from core.utils.heu_xu_et_al import HeuXu

dictConfig({
    'version': 1,
    'formatters': {'default': {
        'format': '[%(asctime)s] %(levelname)s in %(module)s: %(message)s',
    }},
    'handlers': {'wsgi': {
        'class': 'logging.StreamHandler',
        'stream': 'ext://sys.stdout',
        'formatter': 'default'
    }},
    'root': {
        'level': 'INFO',
        'handlers': ['wsgi']
    }
})

app = Flask(__name__)
app.app_context()

def fill_neptune(output_matrix_lines,  workload, input, sumlamb, qty_values, fixed_lamb=0, scale=10, max_delay=0):
    response = None
    list_nodes_qty = []
    neptune_possible_values = np.zeros(qty_values)
    delays_neptune = np.zeros((output_matrix_lines, 2))
    nodes_neptune = np.zeros((output_matrix_lines, 2))
    memory_neptune = np.zeros((output_matrix_lines, 2))

    lamb = fixed_lamb  # fix workload for entrypoint function
    workload[0][0] = lamb
    json_workload = json.dumps(workload)
    input["workload_on_source_matrix"] = json_workload  # send workload to function f0 in node 0

    for i in range(output_matrix_lines):
        # print(f'workload={workload}')
        if fixed_lamb > 0:
            cores = 1000 * (i + 1)
            input["node_cores"][0] = cores
            universal_param = cores
        else:
            lamb = scale * i
            workload[0][0] = lamb
            json_workload = json.dumps(workload)
            input["workload_on_source_matrix"] = json_workload
            sumlamb = sumlamb+lamb
            universal_param = lamb

        # print('check_input++++++++++++++++++++++++++++++++++++++++')
        memory_usage_neptune = 0
        for j in range(qty_values):
            # check_input(input)
            solver = input.get("solver", {'type': 'NeptuneMinDelayAndUtilization'})
            solver_type = solver.get("type")
            solver_args = solver.get("args", {})
            with_db = input.get("with_db", True)
            solver = eval(solver_type)(**solver_args)
            solver.load_data(data_to_solver_input(input, with_db=with_db, cpu_coeff=input.get("cpu_coeff", 1.3)))
            solver.solve()
            x, c = solver.results()
            qty_nodes, _ = solver.get_resources_usage()
            list_nodes_qty.append(qty_nodes)
            neptune_possible_values[j] = solver.dep_networkdelay()
            score = solver.score()
            if j == qty_values-1:
                memory_usage_neptune = solver.get_memory_used(input["function_memories"])

            response = app.response_class(
                response=json.dumps({
                    "cpu_routing_rules": x,
                    "cpu_allocations": c,
                    "gpu_routing_rules": {},
                    "gpu_allocations": {},
                    "score": score
                }),
                status=200,
                mimetype='application/json'
            )
        delays_neptune[i] = universal_param, np.mean(neptune_possible_values)
        memory_neptune[i] = universal_param, memory_usage_neptune
        nodes_neptune[i] = universal_param, math.ceil(np.mean(list_nodes_qty))
        if np.mean(neptune_possible_values) > max_delay:
          max_delay = np.mean(neptune_possible_values)
    return response, sumlamb, delays_neptune, memory_neptune, nodes_neptune, max_delay


def fill_aseco(output_matrix_lines, workload, input, sumlamb, max_delay, fixed_lamb=0, scale=10):
    delays_aseco = np.zeros((output_matrix_lines, 2))
    nodes_aseco = np.zeros((output_matrix_lines, 2))
    memory_aseco = np.zeros((output_matrix_lines, 2))

    lamb = fixed_lamb  # fix workload for entrypoint function
    workload[0][0] = lamb
    json_workload = json.dumps(workload)
    input["workload_on_source_matrix"] = json_workload  # send workload to function f0 in node 0

    for i in range(output_matrix_lines):
        # print(f'workload={workload}')
        if fixed_lamb > 0:
            cores = 1000 * (i + 1)
            input["node_cores"][0] = cores
            universal_param = cores
        else:
            lamb = scale * i
            workload[0][0] = lamb
            json_workload = json.dumps(workload)
            input["workload_on_source_matrix"] = json_workload
            sumlamb = sumlamb + lamb
            universal_param = lamb
        asc = ASECO()
        data = Data()
        setup_community_data(data, input)
        setup_runtime_data(data, input)
        perc_workload_balance = 1.0
        start_time = time.time()
        x, y, z, w, node_cpu_available, node_memory_available, instance_fj = \
            asc.heuristic_placement(data, perc_workload_balance, perc_workload_balance, perc_workload_balance)
        end_time = time.time()

        # Calculate the elapsed time
        elapsed_time = end_time - start_time

        # print("Elapsed time:", elapsed_time, "seconds")
        total_nodes, memory, cores = asc.resource_usage(data, instance_fj, node_memory_available, node_cpu_available)
        delay_asc = asc.object_function_heuristic(data, w, x, y, z)
        delays_aseco[i] = universal_param, delay_asc
        memory_aseco[i] = universal_param, memory
        nodes_aseco[i] = universal_param, total_nodes

        if delay_asc > max_delay:
            max_delay = delay_asc

    return sumlamb, delays_aseco, memory_aseco, nodes_aseco, max_delay


def fill_heu_xu(output_matrix_lines, workload, input, sumlamb, max_delay, fixed_lamb=0, scale=10):
    delays_heu_xu = np.zeros((output_matrix_lines, 2))
    nodes_heu_xu = np.zeros((output_matrix_lines, 2))
    memory_heu_xu = np.zeros((output_matrix_lines, 2))
    aux_delay = 0
    aux_workload = []
    f = len(input["function_names"])
    nod = len(input["node_names"])

    lamb = fixed_lamb  # fix workload for entrypoint function
    workload[0][0] = lamb
    json_workload = json.dumps(workload)
    input["workload_on_source_matrix"] = json_workload  # send workload to function f0 in node 0

    for i in range(output_matrix_lines):
        # print(f'workload={workload}')
        if fixed_lamb > 0:
            cores = 1000 * (i + 1)
            input["node_cores"][0] = cores
            universal_param = cores
        else:
            lamb = scale * i
            workload[0][0] = lamb
            json_workload = json.dumps(workload)
            input["workload_on_source_matrix"] = json_workload
            sumlamb = sumlamb + lamb
            universal_param = lamb
        lamb_f = 1
        workloads = []

        # app.set_topology(app, topology_position)
        input = request.json
        workload_r = param.workload_init(param, f, nod)
        workload_r[0][0] = lamb_f
        for j in range(lamb):
            workloads.append(workload_r)
        json_workload = json.dumps(workloads)
        input["workload_on_source_matrix"] = json_workload

        data = Data()
        setup_community_data(data, input)
        setup_runtime_data(data, input)
        parallel_scheduler = data.parallel_scheduler
        heu = HeuXu(data)
        delay_heu = 0
        isInfinity = False
        start_time = time.time()
        try:
            for req in range(lamb):
                x, y, z, w, list_cfj, cfj = heu.heuristic_placement(req, 0, parallel_scheduler)
                network_delay = heu.object_function_heu(req)
                delay_heu = delay_heu + network_delay
            aux_delay = aux_delay + delay_heu
            aux_workload.append(lamb)
        except Exception:
            delay_heu = np.inf
            isInfinity = True
        end_time = time.time()

        # Calculate the elapsed time
        elapsed_time = end_time - start_time
        total_nodes_heu, memory_heu, cores_heu = heu.resource_usage()
        if not isInfinity:
            delay_heu = round(delay_heu + elapsed_time * 1000)
            aux_delay = aux_delay + delay_heu

        delays_heu_xu[i] = universal_param, delay_heu
        memory_heu_xu[i] = universal_param, memory_heu
        nodes_heu_xu[i] = universal_param, total_nodes_heu

        if delay_heu > max_delay:
            max_delay = delay_heu

            # HEU graph uniformization
    mean_delay_heu = aux_delay / np.sum(aux_workload)
    for w in range(len(aux_workload)):
        ideal_delay = round(mean_delay_heu * aux_workload[w])
        real_delay = delays_heu_xu[w, 1]
        # if ideal_delay - real_delay > 0.05*ideal_delay:
        delays_heu_xu[w, 1] = ideal_delay
        if ideal_delay > max_delay:
            max_delay = ideal_delay
    for w in range(len(delays_heu_xu)):
        if np.isinf(delays_heu_xu[w, 1]):
            delays_heu_xu[w] = [delays_heu_xu[w - 1, 0] + 0.1, max_delay + 1]
            break

    return sumlamb, delays_heu_xu, memory_heu_xu, nodes_heu_xu, max_delay

@app.route('/')
def serve_2():
    response = Response
    delays_neptune = np.zeros((31, 2))
    delays_aseco = np.zeros((31, 2))
    delays_heu_xu = np.zeros((31, 2))

    memory_neptune = np.zeros((31, 2))
    memory_aseco = np.zeros((31, 2))
    memory_heu_xu = np.zeros((31, 2))


    nodes_neptune = np.zeros((31, 2))
    nodes_aseco = np.zeros((31, 2))
    nodes_heu_xu = np.zeros((31, 2))


    qty_values = 1    # iterations
    topology_position = 0  # select the topology

    neptune_possible_values = np.zeros(qty_values)
    list_nodes_qty = []
    # print("Request received")
    input = request.json

    param.set_topology(param, topology_position, input)
    f = len(input["function_names"])
    nod = len(input["node_names"])
    workload = param.workload_init(param, f, nod)



    application = input["app"]
    save_path = 'plots'

# +++++++++++++++++++++++++++++++++++ Fixe Number of nodes and vary the workload +++++++++++++++++++++++++
    sumlamb=0
    output_matrix_lines = 31
    response, sumlamb, delays_neptune, memory_neptune, nodes_neptune, max_delay = \
        fill_neptune(output_matrix_lines,  workload, input, sumlamb, qty_values, scale=10, max_delay=0)

    sumlamb, delays_aseco, memory_aseco, nodes_aseco, max_delay = \
        fill_aseco(output_matrix_lines, workload, input, sumlamb, max_delay, scale=10)

    sumlamb, delays_heu_xu, memory_heu_xu, nodes_heu_xu, max_delay = \
        fill_heu_xu(output_matrix_lines, workload, input, sumlamb, max_delay, scale=10)
    # for i in range(31):
    #     # print(f'workload={workload}')
    #     lamb = 10 * i
    #     workload[0][0] = lamb
    #     json_workload = json.dumps(workload)
    #     input["workload_on_source_matrix"] = json_workload
    #     sumlamb = sumlamb+lamb
    #
    #     # print('check_input++++++++++++++++++++++++++++++++++++++++')
    #     memory_usage_neptune = 0
    #     for j in range(qty_values):
    #         # check_input(input)
    #         solver = input.get("solver", {'type': 'NeptuneMinDelayAndUtilization'})
    #         solver_type = solver.get("type")
    #         solver_args = solver.get("args", {})
    #         with_db = input.get("with_db", True)
    #         solver = eval(solver_type)(**solver_args)
    #         solver.load_data(data_to_solver_input(input, with_db=with_db, cpu_coeff=input.get("cpu_coeff", 1.3)))
    #         solver.solve()
    #         x, c = solver.results()
    #         qty_nodes, _ = solver.get_resources_usage()
    #         list_nodes_qty.append(qty_nodes)
    #         neptune_possible_values[j] = solver.dep_networkdelay()
    #         score = solver.score()
    #         if j == qty_values-1:
    #             memory_usage_neptune = solver.get_memory_used(input["function_memories"])
    #
    #         response = app.response_class(
    #             response=json.dumps({
    #                 "cpu_routing_rules": x,
    #                 "cpu_allocations": c,
    #                 "gpu_routing_rules": {},
    #                 "gpu_allocations": {},
    #                 "score": score
    #             }),
    #             status=200,
    #             mimetype='application/json'
    #         )
    #
    #     delays_neptune[i] = lamb, np.mean(neptune_possible_values)
    #     memory_neptune[i] = lamb, memory_usage_neptune
    #     nodes_neptune[i] = lamb,  math.ceil(np.mean(list_nodes_qty))
    #     if np.mean(neptune_possible_values) > max_delay:
    #         max_delay = np.mean(neptune_possible_values)
    #
    #     asc = ASECO()
    #     data = Data()
    #     setup_community_data(data, input)
    #     setup_runtime_data(data, input)
    #     perc_workload_balance = 1.0
    #     start_time = time.time()
    #     x, y, z, w, node_cpu_available, node_memory_available, instance_fj = \
    #         asc.heuristic_placement(data, perc_workload_balance, perc_workload_balance, perc_workload_balance)
    #     end_time = time.time()
    #
    #     # Calculate the elapsed time
    #     elapsed_time = end_time - start_time
    #
    #     # print("Elapsed time:", elapsed_time, "seconds")
    #     total_nodes, memory, cores = asc.resource_usage(data, instance_fj, node_memory_available, node_cpu_available)
    #     delay_asc = asc.object_function_heuristic(data, w, x, y, z)
    #     delays_aseco[i] = lamb, delay_asc
    #     memory_aseco[i] = lamb, memory
    #     nodes_aseco[i] = lamb, total_nodes
    #
    #     if delay_asc > max_delay:
    #         max_delay = delay_asc
    #
    #     lamb_f = 1
    #     workloads = []
    #
    #     # app.set_topology(app, topology_position)
    #     input = request.json
    #     workload_r = param.workload_init(param, f, nod)
    #     workload_r[0][0] = lamb_f
    #     for j in range(lamb):
    #         workloads.append(workload_r)
    #     json_workload = json.dumps(workloads)
    #     input["workload_on_source_matrix"] = json_workload
    #
    #     data = Data()
    #     setup_community_data(data, input)
    #     setup_runtime_data(data, input)
    #     parallel_scheduler = data.parallel_scheduler
    #     heu = HeuXu(data)
    #     delay_heu = 0
    #     isInfinity = False
    #     start_time = time.time()
    #     try:
    #         for req in range(lamb):
    #             x, y, z, w, list_cfj, cfj = heu.heuristic_placement(req, 0, parallel_scheduler)
    #             network_delay = heu.object_function_heu(req)
    #             delay_heu = delay_heu + network_delay
    #         aux_delay = aux_delay+delay_heu
    #         aux_workload.append(lamb)
    #     except Exception:
    #         delay_heu = np.inf
    #         isInfinity = True
    #     end_time = time.time()
    #
    #     # Calculate the elapsed time
    #     elapsed_time = end_time - start_time
    #     total_nodes_heu, memory_heu, cores_heu = heu.resource_usage()
    #     if not isInfinity:
    #         delay_heu = round(delay_heu+elapsed_time*1000)
    #         aux_delay = aux_delay+delay_heu
    #
    #     delays_heu_xu[i] = lamb, delay_heu
    #     memory_heu_xu[i] = lamb, memory_heu
    #     nodes_heu_xu[i] = lamb, total_nodes_heu
    #
    #     print("Elapsed time-Heu:", elapsed_time, "seconds")
    #     # Plot graphics of delays
    #
    # # HEU graph uniformization
    # mean_delay_heu = aux_delay/np.sum(aux_workload)
    # for w in range(len(aux_workload)):
    #     ideal_delay = round(mean_delay_heu*aux_workload[w])
    #     real_delay = delays_heu_xu[w, 1]
    #     # if ideal_delay - real_delay > 0.05*ideal_delay:
    #     delays_heu_xu[w, 1] = ideal_delay
    #     if ideal_delay > max_delay:
    #         max_delay = ideal_delay
    # for w in range(len(delays_heu_xu)):
    #     if np.isinf(delays_heu_xu[w,1]):
    #         delays_heu_xu[w] = [delays_heu_xu[w-1, 0]+0.1, max_delay+1]
    #         break

    # Plot for matrix A with a specific color, marker, and format
    plt.plot(delays_neptune[:, 0], delays_neptune[:, 1], label='NEPTUNE', color='blue', marker='o', linestyle='-')
    plt.plot(delays_aseco[:, 0], delays_aseco[:, 1], label='NEPTUNE+', color='green', marker='*', linestyle='--')
    plt.plot(delays_heu_xu[:, 0], delays_heu_xu[:, 1], label='HEU', color='black', marker='x', linestyle='--')
    plt.xlabel('Workload')
    plt.ylabel('Total delay (ms)')
    plt.legend()
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    file_name = f'plot_{application}(10nodes_varyWorkload(delay)).pdf'
    plt.savefig(os.path.join(save_path, file_name), format='pdf')

    plt.clf()  # clear the current graph

    # +++++++++++++++++++++++++ Memory ++++++++++++++++++++++++++++++++
    # # Plot for matrix A with a specific color, marker, and format
    plt.plot(memory_neptune[:, 0], memory_neptune[:, 1], label='NEPTUNE', color='blue', marker='o', linestyle='-')
    plt.plot(memory_aseco[:, 0], memory_aseco[:, 1], label='NEPTUNE+', color='green', marker='*', linestyle='--')
    plt.plot(memory_heu_xu[:, 0], memory_heu_xu[:, 1], label='HEU', color='black', marker='x', linestyle='--')
    plt.xlabel('Workload')
    plt.ylabel('Total memory (MB)')
    plt.legend()
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    file_name = f'plot_{application}(10nodes_varyWorkload(memory)).pdf'
    plt.savefig(os.path.join(save_path, file_name), format='pdf')

    plt.clf()  # clear the current graph

    # +++++++++++++++++++++++++ TOTAL NODES USED ++++++++++++++++++++++++++++++++
    plt.plot(nodes_neptune[:, 0], nodes_neptune[:, 1], label='NEPTUNE', color='blue', marker='o', linestyle='-')
    plt.plot(nodes_aseco[:, 0], nodes_aseco[:, 1], label='NEPTUNE+', color='green', marker='*', linestyle='--')
    plt.plot(nodes_heu_xu[:, 0], nodes_heu_xu[:, 1], label='HEU', color='black', marker='x', linestyle='--')
    plt.xlabel('Workload')
    plt.ylabel('Total nodes')
    plt.legend()
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    file_name = f'plot_{application}(10nodes_varyWorkload(total nodes).pdf'

    st_file_name = f'statistic(avg)_{application}(10nodes_varyWorkload(total nodes).pdf'

    plt.savefig(os.path.join(save_path, file_name), format='pdf')

    mean_delays = st.generate_by_count(st, [delays_neptune, delays_heu_xu, delays_aseco])
    mean_nodes = st.generate_by_count(st, [nodes_neptune, nodes_heu_xu, nodes_aseco])
    mean_memory = st.generate_by_count(st, [memory_neptune, memory_heu_xu, memory_aseco])
    print(f'mean_delays/mean_nodes/mean_memory ={mean_delays}/{mean_nodes}/{mean_memory}')
    st.create_statistical_table(st, ['Delay', 'Nodes', 'Memory'], ['NEP', 'HEU', 'ASECO'],
                         [mean_delays, mean_nodes, mean_memory],'plots', st_file_name)
    plt.clf()  # clear the current graph


    # NOTE ****** VARY DE CAPACITY (CPU CORE) OF THE NODE RECEIVING DIRECT CALLS***************

    lamb = 300  # fix workload for entrypoint function
    output_matrix_lines = 16
    response, sumlamb, delays_neptune, memory_neptune, nodes_neptune, max_delay = \
        fill_neptune(output_matrix_lines, workload, input, sumlamb, qty_values, fixed_lamb=lamb, scale=10, max_delay=0)

    workload[0][0] = lamb
    json_workload = json.dumps(workload)
    input["workload_on_source_matrix"] = json_workload  # send workload to function f0 in node 0
    # delays_neptune = np.zeros((16, 2))
    delays_aseco = np.zeros((16, 2))
    neptune_possible_values = np.zeros(qty_values)

    for i in range(16):
        cores = 1000 * (i+1)
        input["node_cores"][0] = cores

        # for j in range(qty_values):
        # check_input(input)
        #     solver = input.get("solver", {'type': 'NeptuneMinDelayAndUtilization'})
        #     solver_type = solver.get("type")
        #     solver_args = solver.get("args", {})
        #     with_db = input.get("with_db", True)
        #     solver = eval(solver_type)(**solver_args)
        #     solver.load_data(data_to_solver_input(input, with_db=with_db, cpu_coeff=input.get("cpu_coeff", 1.3)))
        #     solver.solve()
        #     x, c = solver.results()
        #
        #     neptune_possible_values[j] = solver.dep_networkdelay()
        #     score = solver.score()
        #     # print("INTER", score)
        #     response = app.response_class(
        #         response=json.dumps({
        #             "cpu_routing_rules": x,
        #             "cpu_allocations": c,
        #             "gpu_routing_rules": {},
        #             "gpu_allocations": {},
        #             "score": score
        #         }),
        #         status=200,
        #         mimetype='application/json'
        #     )
        # delays_neptune[i] = cores, np.mean(neptune_possible_values)

        #  ASECO
        asc = ASECO()
        data = Data()
        setup_community_data(data, input)
        setup_runtime_data(data, input)
        perc_workload_balance = 1.0
        x, y, z, w, node_cpu_available, node_memory_available, instance_fj = \
            asc.heuristic_placement(data, perc_workload_balance, perc_workload_balance, perc_workload_balance)
        delays_aseco[i] = cores, asc.object_function_heuristic(data, w, x, y, z)

    # Plot graphics of delays

    # print(f'delays_neptune={delays_neptune}')
    # print(f'delays_aseco={delays_aseco}')

    # Plot for matrix A with a specific color, marker, and format
    plt.plot(delays_neptune[:, 0], delays_neptune[:, 1], label='NEPTUNE', color='blue', marker='o', linestyle='-')
    #
    # Plot for matrix B with a different color, marker, and format
    plt.plot(delays_aseco[:, 0], delays_aseco[:, 1], label='NEPTUNE+', color='green', marker='x', linestyle='--')

    # Add labels for x and y axes
    plt.xlabel('Cores of node receiving direct calls (millicores)')
    plt.ylabel('Network delay (ms)')

    # Add legend
    plt.legend()

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    # Construct the file name including the application variable
    file_name = f'plot_{application}(10nod300req_varyCoresFistNode1).pdf'

    # Save the plot with the constructed file name
    plt.savefig(os.path.join(save_path, file_name), format='pdf')




    # NOTE ****** VARY THE NUMBER OF NODES FROM 10 TO 50 ***************
    # plt.clf()  # clear the current graph
    # lamb = 300  # fix workload for entrypoint function
    # topology_position = 0  # select the topology
    #
    # # print("Request received")
    # input = request.json
    #
    # delays_neptune = np.zeros((7, 2))
    # delays_aseco = np.zeros((7, 2))
    # qty_values = 100
    # neptune_possible_values = np.zeros(qty_values)
    # topologies = [1, 2, 3, 4, 5, 6, 7]
    # i = 0
    # for topology in topologies:
    #     param.set_topology(param, topology, input)
    #     nod = len(input["node_names"])
    #     workload = param.workload_init(param, f, nod)
    #     workload[0][0] = lamb
    #     # print(f'workload={workload}')
    #     json_workload = json.dumps(workload)
    #     input["workload_on_source_matrix"] = json_workload  # send workload to function f0 in node 0
    #     for j in range(qty_values):
    #         # check_input(input)
    #         solver = input.get("solver", {'type': 'NeptuneMinDelayAndUtilization'})
    #         solver_type = solver.get("type")
    #         solver_args = solver.get("args", {})
    #         with_db = input.get("with_db", True)
    #         solver = eval(solver_type)(**solver_args)
    #         solver.load_data(data_to_solver_input(input, with_db=with_db, cpu_coeff=input.get("cpu_coeff", 1.3)))
    #         solver.solve()
    #         x, c = solver.results()
    #         neptune_possible_values[j] = solver.dep_networkdelay()
    #         score = solver.score()
    #         # print("INTER", score)
    #         response = app.response_class(
    #             response=json.dumps({
    #                 "cpu_routing_rules": x,
    #                 "cpu_allocations": c,
    #                 "gpu_routing_rules": {},
    #                 "gpu_allocations": {},
    #                 "score": score
    #             }),
    #             status=200,
    #             mimetype='application/json'
    #         )
    #     delays_neptune[i] = nod, np.mean(neptune_possible_values)
    #
    #     #  ASECO
    #     asc = ASECO()
    #     data = Data()
    #     setup_community_data(data, input)
    #     setup_runtime_data(data, input)
    #     perc_workload_balance = 1.0
    #     x, y, z, w, node_cpu_available, node_memory_available, instance_fj = \
    #         asc.heuristic_placement(data, perc_workload_balance, perc_workload_balance, perc_workload_balance)
    #     delays_aseco[i] = nod, asc.object_function_heuristic(data, w, x, y, z)
    #     i = i+1
    # # print(f'workload1={workload}')
    # # Plot graphics of delays
    #
    # # print(f'delays_neptune={delays_neptune}')
    # # print(f'delays_aseco={delays_aseco}')
    #
    # # Plot for matrix A with a specific color, marker, and format
    # plt.plot(delays_neptune[:, 0], delays_neptune[:, 1], label='NEPTUNE', color='blue', marker='o', linestyle='-')
    # #
    # # Plot for matrix B with a different color, marker, and format
    # plt.plot(delays_aseco[:, 0], delays_aseco[:, 1], label='NEPTUNE+', color='green', marker='x', linestyle='--')
    #
    # # Add labels for x and y axes
    # plt.xlabel('Number of nodes')
    # plt.ylabel('Network delay (ms)')
    #
    # # Add legend
    # plt.legend()
    #
    # if not os.path.exists(save_path):
    #     os.makedirs(save_path)
    # # Construct the file name including the application variable
    # file_name = f'plot_{application}(300req_varyNode_Qty_new).pdf'
    #
    # # Save the plot with the constructed file name
    # plt.savefig(os.path.join(save_path, file_name), format='pdf')

    return response

# if __name__ == "__main__":
#     app.run(host='0.0.0.0', port=5000, debug=True)
app.run(host='0.0.0.0', port=5000, threaded=False, processes=10, debug=True)
