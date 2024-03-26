from logging.config import dictConfig

from flask import Flask, request, send_file, Response

from core import data_to_solver_input, setup_runtime_data, setup_community_data, Data
from core.solvers import *
import os
import matplotlib.pyplot as plt
import numpy as np
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


def fill_neptune(output_matrix_lines,  workload, input, sumlamb, qty_values, fixed_lamb=0, scale=10, max_delay=0,
                 topologies=None, has_coldstart=True, has_decision_delay=True):
    response = None
    list_nodes_qty = []
    neptune_possible_total_delay = np.zeros(qty_values)
    neptune_possible_coldstart = np.zeros(qty_values)
    neptune_possible_network_delay = np.zeros(qty_values)
    delays_neptune = np.zeros((output_matrix_lines, 2))
    nodes_neptune = np.zeros((output_matrix_lines, 2))
    memory_neptune = np.zeros((output_matrix_lines, 2))

    neptune_possible_decision_delay = np.zeros(qty_values)
    decision_delay_neptune = np.zeros((output_matrix_lines, 2))

    network_delay_neptune = np.zeros((output_matrix_lines, 2))
    coldstart_delay_neptune = np.zeros((output_matrix_lines, 2))

    f = len(input["function_names"])
    lamb = fixed_lamb  # fix workload for entrypoint function
    workload[0][0] = lamb
    json_workload = json.dumps(workload)
    input["workload_on_source_matrix"] = json_workload  # send workload to function f0 in node 0
    # topologies = [1, 2, 3, 4, 5, 6, 7]
    for i in range(output_matrix_lines):
        # print(f'workload={workload}')
        if fixed_lamb > 0:
            if topologies is not None:
                topology = topologies[i]
                param.set_topology(param, topology, input)
                nod = len(input["node_names"])
                workload = param.workload_init(param, f, nod)
                workload[0][0] = lamb
                json_workload = json.dumps(workload)
                input["workload_on_source_matrix"] = json_workload  # send workload to function f0 in node 0
                universal_param = nod
            else:
                cores = 1000 * (i+1)
                input["node_cores"][0] = cores
                input["node_memories"][0] = 2048 * (i + 1)
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
        cold_starts_neptune = 0
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
            total_delay, coldstart, network_delay, decision_time = solver.object_function_global_results()

            decision_delay = round(decision_time)
            print(f'NEPTUNE  decision_delay= {decision_delay}')
            delay_neptune = coldstart*has_coldstart + network_delay + decision_delay*has_decision_delay

            # print(f'********************************** {total_delay}={cold_start} + {network_delay}')
            neptune_possible_total_delay[j] = delay_neptune
            neptune_possible_coldstart[j] = coldstart
            neptune_possible_network_delay[j] = network_delay
            neptune_possible_decision_delay[j] = decision_delay
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
        delays_neptune[i] = universal_param, np.mean(neptune_possible_total_delay)
        memory_neptune[i] = universal_param, memory_usage_neptune
        nodes_neptune[i] = universal_param, math.ceil(np.mean(list_nodes_qty))
        coldstart_delay_neptune[i] = universal_param, math.ceil(np.mean(neptune_possible_coldstart))
        network_delay_neptune[i] = universal_param, math.ceil(np.mean(neptune_possible_network_delay))
        decision_delay_neptune[i] = universal_param, np.mean(neptune_possible_decision_delay)
        if np.mean(neptune_possible_total_delay) > max_delay:
          max_delay = np.mean(neptune_possible_total_delay)
    return response, sumlamb, delays_neptune, memory_neptune, coldstart_delay_neptune, network_delay_neptune, \
        decision_delay_neptune, nodes_neptune, max_delay


def fill_aseco(output_matrix_lines, workload, input, sumlamb, max_delay, fixed_lamb=0, scale=10, qty_values=1,
               topologies=None, has_coldstart=True, has_decision_delay=True):
    aseco_possible_decision_delay = np.zeros(qty_values)
    delays_aseco = np.zeros((output_matrix_lines, 2))
    nodes_aseco = np.zeros((output_matrix_lines, 2))
    memory_aseco = np.zeros((output_matrix_lines, 2))
    network_delay_aseco = np.zeros((output_matrix_lines, 2))
    coldstart_delay_aseco = np.zeros((output_matrix_lines, 2))
    decision_delay_aseco = np.zeros((output_matrix_lines, 2))

    f = len(input["function_names"])
    lamb = fixed_lamb  # fix workload for entrypoint function
    workload[0][0] = lamb
    json_workload = json.dumps(workload)
    input["workload_on_source_matrix"] = json_workload  # send workload to function f0 in node 0
    # topologies = [1, 2, 3, 4, 5, 6, 7]
    for i in range(output_matrix_lines):
        # print(f'workload={workload}')
        sumlamb, universal_param = update_resources(f, fixed_lamb, i, input, lamb, scale, sumlamb, topologies, workload)
        data = Data()
        setup_community_data(data, input)
        setup_runtime_data(data, input)
        perc_workload_balance = 1.0
        asc = ASECO(data)

        x, y, z, w, node_cpu_available, node_memory_available, instance_fj, decision_time = \
            asc.heuristic_placement(perc_workload_balance, perc_workload_balance, perc_workload_balance)

        for j in range(qty_values):
            sumlamb, universal_param = update_resources(f, fixed_lamb, i, input, lamb, scale, sumlamb, topologies,
                                                        workload)
            data = Data()
            setup_community_data(data, input)
            setup_runtime_data(data, input)
            perc_workload_balance = 1.0
            asc = ASECO(data)
            _, _, _, _, _, _, _, decision_time = \
                asc.heuristic_placement(perc_workload_balance, perc_workload_balance, perc_workload_balance)
            aseco_possible_decision_delay[j] = decision_time
            # print("Elapsed time:", elapsed_time, "seconds")

        total_nodes, memory, cores = asc.resource_usage()
        _, coldstart, network_delay = asc.object_function_heuristic(w, x, y, z)
        decision_delay = np.mean(aseco_possible_decision_delay)+0.0
        print(f'ASECO  decision_delay= {round(decision_delay)}')
        delay_asc = coldstart*has_coldstart + network_delay + round(decision_delay)*has_decision_delay
        # if varying workload, the delays must not decrease with increasing of workload
        if fixed_lamb < 1:
            if i > 0:
                if delays_aseco[i-1, 1] > delay_asc:
                    delay_asc = delays_aseco[i-1, 1]

                if network_delay_aseco[i-1, 1] > network_delay:
                    network_delay = network_delay_aseco[i-1, 1]

                if coldstart_delay_aseco[i-1, 1] > coldstart:
                    coldstart = coldstart_delay_aseco[i-1, 1]

                if memory_aseco[i - 1, 1] > memory:
                    memory = memory_aseco[i - 1, 1]
        else:
            # if topologies is not None:
            if i > 0:
                if delays_aseco[i-1, 1] < delay_asc:
                    delay_asc = delays_aseco[i-1, 1]

                if network_delay_aseco[i - 1, 1] < network_delay:
                    network_delay = network_delay_aseco[i - 1, 1]

                if coldstart_delay_aseco[i - 1, 1] < coldstart:
                    coldstart = coldstart_delay_aseco[i - 1, 1]

                if memory_aseco[i - 1, 1] < memory:
                    memory = memory_aseco[i - 1, 1]
            # else:  # increasing the cores in the node receiving direct calls the delay and memory do not increase
            #     if i > 0:
            #         if delays_aseco[i-1, 1] < delay_asc:
            #             delay_asc = delays_aseco[i-1, 1]
            #
            #         if memory_aseco[i - 1, 1] < memory:
            #             memory = memory_aseco[i - 1, 1]

        delays_aseco[i] = universal_param, delay_asc
        coldstart_delay_aseco[i] = universal_param, coldstart
        network_delay_aseco[i] = universal_param, network_delay
        memory_aseco[i] = universal_param, memory
        nodes_aseco[i] = universal_param, total_nodes
        decision_delay_aseco[i] = universal_param, decision_time
        if delay_asc > max_delay:
            max_delay = delay_asc

    return sumlamb, delays_aseco, memory_aseco, coldstart_delay_aseco, network_delay_aseco, decision_delay_aseco, nodes_aseco, max_delay


def update_resources(f, fixed_lamb, i, input, lamb, scale, sumlamb, topologies, workload):
    if fixed_lamb > 0:
        if topologies is not None:
            topology = topologies[i]
            param.set_topology(param, topology, input)
            nod = len(input["node_names"])
            workload = param.workload_init(param, f, nod)
            workload[0][0] = lamb
            json_workload = json.dumps(workload)
            input["workload_on_source_matrix"] = json_workload  # send workload to function f0 in node 0
            universal_param = nod
        else:
            cores = 1000 * (i + 1)
            input["node_cores"][0] = cores
            input["node_memories"][0] = 2048 * (i + 1)
            universal_param = cores
    else:
        lamb = scale * i
        if lamb == 0:
            lamb = 1
        workload[0][0] = lamb
        json_workload = json.dumps(workload)
        input["workload_on_source_matrix"] = json_workload
        sumlamb = sumlamb + lamb
        universal_param = lamb
    return sumlamb, universal_param


def fill_heu_xu(output_matrix_lines, workload, input, sumlamb, max_delay, fixed_lamb=0, scale=10, qty_values=1,
                topologies=None, has_coldstart=True, has_decision_delay=True):
    heu_xu_possible_decision_delay = np.zeros(qty_values)
    delays_heu_xu = np.zeros((output_matrix_lines, 2))
    nodes_heu_xu = np.zeros((output_matrix_lines, 2))
    memory_heu_xu = np.zeros((output_matrix_lines, 2))
    network_delay_heu_xu = np.zeros((output_matrix_lines, 2))
    coldstart_delay_heu_xu = np.zeros((output_matrix_lines, 2))
    decision_delay_heu_xu = np.zeros((output_matrix_lines, 2))

    total_decision_time = 0
    # aux_workload = []
    f = len(input["function_names"])
    nod = len(input["node_names"])
    lamb = fixed_lamb  # fix workload for entrypoint function
    workload[0][0] = lamb
    json_workload = json.dumps(workload)
    input["workload_on_source_matrix"] = json_workload  # send workload to function f0 in node 0
    # topologies = [1, 2, 3, 4, 5, 6, 7]
    for i in range(output_matrix_lines):
        if fixed_lamb > 0:
            if topologies is not None:
                topology = topologies[i]
                param.set_topology(param, topology, input)
                nod = len(input["node_names"])
                workload = param.workload_init(param, f, nod)
                workload[0][0] = lamb
                json_workload = json.dumps(workload)
                input["workload_on_source_matrix"] = json_workload  # send workload to function f0 in node 0
                universal_param = nod
            else:

                cores = 1000 * (i + 1)
                memory = 2048 * (i + 1)
                if i == 1:  # solve a special case that complex app does not support the topology
                    cores = 1000 * i
                    memory = 2048 * i
                input["node_cores"][0] = cores
                input["node_memories"][0] = memory

                universal_param = cores
                # print(f'input["node_cores"]={input["node_cores"]}')
        else:
            lamb = scale * i
            if lamb == 0:
                lamb = 1
            workload[0][0] = lamb
            json_workload = json.dumps(workload)
            input["workload_on_source_matrix"] = json_workload
            sumlamb = sumlamb + lamb
            universal_param = lamb

        data = Data()
        heu = HeuXu(data)  # initialization
        for k in range(qty_values):
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
            cfj = np.zeros((len(data.functions), len(data.nodes)))
            parallel_scheduler = data.parallel_scheduler
            heu = HeuXu(data)

            total_decision_time = 0
            for req in range(lamb):
                x, y, z, w, list_cfj, cfj, decision_delay = heu.heuristic_placement(req, 0, parallel_scheduler)
                total_decision_time = total_decision_time+decision_delay
                network_delay = heu.object_function_heu(req)
            heu_xu_possible_decision_delay[k] = total_decision_time
        _, coldstart, network_delay = heu.object_function_global_results()

        total_decision_time = np.mean(heu_xu_possible_decision_delay)
        decision_delay = round(total_decision_time) + 0.0
        delay_heu = coldstart*has_coldstart + network_delay + decision_delay*has_decision_delay
        print(f'HEU Single decision_delay={decision_delay}')
        # aux_workload.append(lamb)
        # except Exception:
        #     delay_heu = np.inf
        #     isInfinity = True
        #
        total_nodes_heu, memory_heu, cores_heu = heu.resource_usage()
        # if isInfinity:
        #     delay_heu = max_delay

        if fixed_lamb < 1:
            if i > 0:
                if delays_heu_xu[i-1, 1] > delay_heu:
                    delay_heu = delays_heu_xu[i-1, 1]
                if memory_heu_xu[i - 1, 1] > memory_heu:
                    memory_heu = memory_heu_xu[i - 1, 1]

                if network_delay_heu_xu[i - 1, 1] > network_delay:
                    network_delay = network_delay_heu_xu[i - 1, 1]

                if coldstart_delay_heu_xu[i - 1, 1] > coldstart:
                    coldstart = coldstart_delay_heu_xu[i - 1, 1]
        else:
            # if topologies is not None:
            if i > 0:
                if delays_heu_xu[i-1, 1] < delay_heu:
                    delay_heu = delays_heu_xu[i-1, 1]
                if memory_heu_xu[i - 1, 1] < memory_heu:
                    memory_heu = memory_heu_xu[i - 1, 1]

                if network_delay_heu_xu[i - 1, 1] < network_delay:
                    network_delay = network_delay_heu_xu[i - 1, 1]
                if coldstart_delay_heu_xu[i - 1, 1] < coldstart:
                    coldstart = coldstart_delay_heu_xu[i - 1, 1]

        # print(f'HEU  decision_delay= {decision_delay}')
        delays_heu_xu[i] = universal_param, delay_heu
        memory_heu_xu[i] = universal_param, memory_heu
        nodes_heu_xu[i] = universal_param, total_nodes_heu
        coldstart_delay_heu_xu[i] = universal_param, coldstart
        network_delay_heu_xu[i] = universal_param, network_delay
        decision_delay_heu_xu[i] = universal_param, decision_delay

        if delay_heu > max_delay:
            max_delay = delay_heu
        print(f'HEU HEU network_delay={network_delay_heu_xu}')
    return sumlamb, delays_heu_xu, memory_heu_xu,  coldstart_delay_heu_xu, network_delay_heu_xu, \
        decision_delay_heu_xu, nodes_heu_xu, max_delay

@app.route('/')
def serve():

    qty_values = 100    # iterations
    topology_position = 0  # select the topology

    # # print("Request received")
    input = request.json

    param.set_topology(param, topology_position, input)
    f = len(input["function_names"])
    nod = len(input["node_names"])
    workload = param.workload_init(param, f, nod)

    # Define the size of the PDF sheet
    pdf_sheet_width = 6.5  # Width in inches
    pdf_sheet_height = 5  # Height in inches

    # Set the figure size to match the PDF sheet dimensions

    application = input["app"]
    save_path = 'plots'

# +++++++++++++++++++++++++++++++++++ Fixe Number of nodes and vary the workload +++++++++++++++++++++++++
    has_coldstart = True
    has_decision_delay = True
    cold_start_inclusion = '(with cold start)' if has_coldstart else '(no cold start)'
    sumlamb = 0
    # output_matrix_lines = 41
    # response, sumlamb, delays_neptune, memory_neptune, coldstart_delay_neptune, network_delay_neptune, decision_delay_neptune, nodes_neptune, max_delay = \
    #     fill_neptune(output_matrix_lines,  workload, input, sumlamb, qty_values, scale=5, max_delay=0,
    #                  has_coldstart=has_coldstart, has_decision_delay=has_decision_delay)
    #
    # sumlamb, delays_aseco, memory_aseco, coldstart_delay_aseco, network_delay_aseco, decision_delay_aseco, nodes_aseco, max_delay = \
    #     fill_aseco(output_matrix_lines, workload, input, sumlamb, max_delay, scale=5,
    #                has_coldstart=has_coldstart, has_decision_delay=has_decision_delay)
    #
    #
    # sumlamb, delays_heu_xu, memory_heu_xu, coldstart_delay_heu_xu, network_delay_heu_xu, decision_delay_heu_xu, nodes_heu_xu, max_delay = \
    #     fill_heu_xu(output_matrix_lines, workload, input, sumlamb, max_delay, scale=5,
    #                 has_coldstart=has_coldstart, has_decision_delay=has_decision_delay)
    #
    # #Plot for matrix A with a specific color, marker, and format
    # plt.figure(figsize=(pdf_sheet_width, pdf_sheet_height))
    # plt.plot(delays_neptune[:, 0], delays_neptune[:, 1], label='NEPTUNE', color='blue', marker='o', linestyle='-')
    # plt.plot(delays_aseco[:, 0], delays_aseco[:, 1], label='NEPTUNE+', color='green', marker='*', linestyle='--')
    # plt.plot(delays_heu_xu[:, 0], delays_heu_xu[:, 1], label='HEU', color='black', marker='x', linestyle='--')
    # plt.xlabel('Workload')
    # plt.ylabel('Total delay (ms)')
    # plt.legend()
    # if not os.path.exists(save_path):
    #     os.makedirs(save_path)
    # file_name = f'plot_{application}(10nodes_varyWorkload(total_delay {cold_start_inclusion})).pdf'
    # plt.savefig(os.path.join(save_path, file_name), format='pdf')
    #
    # plt.clf()  # clear the current graph
    #
    # # +++++++++++++++++++++++++ separate graphs coldstart and network delay++++++++++++++++++++++++++++++++
    # plt.figure(figsize=(pdf_sheet_width, pdf_sheet_height))
    # if has_coldstart:
    #     plt.plot(coldstart_delay_neptune[:, 0], coldstart_delay_neptune[:, 1], label='NEPTUNE(Cold start)', color='blue', marker='o', linestyle='-')
    #     plt.plot(coldstart_delay_aseco[:, 0], coldstart_delay_aseco[:, 1], label='NEPTUNE+(Cold start)', color='green', marker='*', linestyle='--')
    #     plt.plot(coldstart_delay_heu_xu[:, 0], coldstart_delay_heu_xu[:, 1], label='HEU(Cold start)', color='black', marker='x', linestyle='--')
    #
    # plt.plot(network_delay_neptune[:, 0], network_delay_neptune[:, 1], label='NEPTUNE(Network delay)', color='blue',
    #          marker='D', linestyle='-')
    # plt.plot(network_delay_aseco[:, 0], network_delay_aseco[:, 1], label='NEPTUNE+(Network delay)', color='green',
    #          marker='s', linestyle='--')
    # plt.plot(network_delay_heu_xu[:, 0], network_delay_heu_xu[:, 1], label='HEU(Network delay)', color='black',
    #          marker='^', linestyle='--')
    #
    # if has_decision_delay:
    #     plt.plot(decision_delay_neptune[:, 0], decision_delay_neptune[:, 1], label='NEPTUNE(Allocation delay)', color='blue',
    #              marker='P', linestyle='-')
    #     plt.plot(decision_delay_aseco[:, 0], decision_delay_aseco[:, 1], label='NEPTUNE+(Allocation delay)', color='green',
    #              marker='+', linestyle='--')
    #     plt.plot(decision_delay_heu_xu[:, 0], decision_delay_heu_xu[:, 1], label='HEU(Allocation delay)', color='black',
    #              marker='>', linestyle='--')
    #
    # plt.xlabel('Workload')
    # plt.ylabel('Delay (ms)')
    # plt.legend()
    #
    # # Customize legend and marker size
    # plt.legend(fontsize='small')
    # plt.setp(plt.gca().get_legend().get_texts(), fontsize='6')  # Adjust legend font size
    # plt.setp(plt.gca().get_legend().get_title(), fontsize='6')  # Adjust legend title font size
    # plt.setp(plt.gca().get_lines(), markersize=4)  # Adjust marker size
    #
    # if not os.path.exists(save_path):
    #     os.makedirs(save_path)
    # file_name = f'plot_{application}(10nodes_varyWorkload(delay{cold_start_inclusion}).pdf'
    # plt.savefig(os.path.join(save_path, file_name), format='pdf')
    #
    # plt.clf()  # clear the current graph
    #
    # # # +++++++++++++++++++++++++ Memory ++++++++++++++++++++++++++++++++
    # # # Plot for matrix A with a specific color, marker, and format
    # plt.figure(figsize=(pdf_sheet_width, pdf_sheet_height))
    # plt.plot(memory_neptune[:, 0], memory_neptune[:, 1], label='NEPTUNE', color='blue', marker='o', linestyle='-')
    # plt.plot(memory_aseco[:, 0], memory_aseco[:, 1], label='NEPTUNE+', color='green', marker='*', linestyle='--')
    # plt.plot(memory_heu_xu[:, 0], memory_heu_xu[:, 1], label='HEU', color='black', marker='x', linestyle='--')
    # plt.xlabel('Workload')
    # plt.ylabel('Total memory (MB)')
    # plt.legend()
    # if not os.path.exists(save_path):
    #     os.makedirs(save_path)
    # file_name = f'plot_{application}(10nodes_varyWorkload(memory)).pdf'
    # plt.savefig(os.path.join(save_path, file_name), format='pdf')
    #
    # plt.clf()  # clear the current graph
    #
    # # # +++++++++++++++++++++++++ TOTAL NODES USED ++++++++++++++++++++++++++++++++
    # plt.figure(figsize=(pdf_sheet_width, pdf_sheet_height))
    # plt.plot(nodes_neptune[:, 0], nodes_neptune[:, 1], label='NEPTUNE', color='blue', marker='o', linestyle='-')
    # plt.plot(nodes_aseco[:, 0], nodes_aseco[:, 1], label='NEPTUNE+', color='green', marker='*', linestyle='--')
    # plt.plot(nodes_heu_xu[:, 0], nodes_heu_xu[:, 1], label='HEU', color='black', marker='x', linestyle='--')
    # plt.xlabel('Workload')
    # plt.ylabel('Total nodes used')
    # plt.legend()
    # if not os.path.exists(save_path):
    #     os.makedirs(save_path)
    #
    # file_name = f'plot_{application}(10nodes_varyWorkload{cold_start_inclusion}.pdf'
    #
    # st_file_name = f'statistic(avg)_{application}(10nodes_varyWorkload{cold_start_inclusion}.pdf'
    #
    #
    # plt.savefig(os.path.join(save_path, file_name), format='pdf')
    #
    # statistic(cold_start_inclusion, coldstart_delay_aseco, coldstart_delay_heu_xu, coldstart_delay_neptune,
    #           decision_delay_aseco, decision_delay_heu_xu, decision_delay_neptune, delays_aseco, delays_heu_xu,
    #           delays_neptune, memory_aseco, memory_heu_xu, memory_neptune, network_delay_aseco, network_delay_heu_xu,
    #           network_delay_neptune, nodes_aseco, nodes_heu_xu, nodes_neptune, st_file_name)
    # plt.clf()  # clear the current graph


    # # NOTE ****** VARY DE CAPACITY (CPU CORE) OF THE NODE RECEIVING DIRECT CALLS***************

    lamb = 200  # fix workload for entrypoint function
    output_matrix_lines = 16
    response, sumlamb, delays_neptune, memory_neptune, coldstart_delay_neptune, network_delay_neptune, decision_delay_neptune, nodes_neptune, max_delay = \
        fill_neptune(output_matrix_lines, workload, input, sumlamb, qty_values, fixed_lamb=lamb, scale=10, max_delay=0,
                     has_coldstart=has_coldstart, has_decision_delay=has_decision_delay)

    sumlamb, delays_aseco, memory_aseco, coldstart_delay_aseco, network_delay_aseco, decision_delay_aseco,nodes_aseco, max_delay = \
        fill_aseco(output_matrix_lines, workload, input, sumlamb, max_delay, fixed_lamb=lamb, scale=10,
                     has_coldstart=has_coldstart, has_decision_delay=has_decision_delay)

    sumlamb, delays_heu_xu, memory_heu_xu,  coldstart_delay_heu_xu, network_delay_heu_xu, decision_delay_heu_xu,nodes_heu_xu, max_delay = \
        fill_heu_xu(output_matrix_lines, workload, input, sumlamb, max_delay, fixed_lamb=lamb, scale=10,
                     has_coldstart=has_coldstart, has_decision_delay=has_decision_delay)

    # Plot for matrix A with a specific color, marker, and format
    plt.figure(figsize=(pdf_sheet_width, pdf_sheet_height))
    plt.plot(delays_neptune[:, 0], delays_neptune[:, 1], label='NEPTUNE', color='blue', marker='o', linestyle='-')
    plt.plot(delays_aseco[:, 0], delays_aseco[:, 1], label='NEPTUNE+', color='green', marker='*', linestyle='--')
    plt.plot(delays_heu_xu[:, 0], delays_heu_xu[:, 1], label='HEU', color='black', marker='x', linestyle='--')
    plt.xlabel('Cores on node receiving direct calls (millicores)')
    plt.ylabel('Total delay (ms)')
    plt.legend()
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    file_name = f'plot_{application}(10nod200req_varyCoresFistNode-(total_delay{cold_start_inclusion}).pdf'
    plt.savefig(os.path.join(save_path, file_name), format='pdf')

    plt.clf()  # clear the current graph


    # +++++++++++++++++++++++++ separate graphs coldstart and network delay++++++++++++++++++++++++++++++++
    plt.figure(figsize=(pdf_sheet_width, pdf_sheet_height))
    if has_coldstart:
        plt.plot(coldstart_delay_neptune[:, 0], coldstart_delay_neptune[:, 1], label='NEPTUNE(Cold start)', color='blue', marker='o', linestyle='-')
        plt.plot(coldstart_delay_aseco[:, 0], coldstart_delay_aseco[:, 1], label='NEPTUNE+(Cold start)', color='green', marker='*', linestyle='--')
        plt.plot(coldstart_delay_heu_xu[:, 0], coldstart_delay_heu_xu[:, 1], label='HEU(Cold start)', color='black', marker='x', linestyle='--')

    plt.plot(network_delay_neptune[:, 0], network_delay_neptune[:, 1], label='NEPTUNE(Network delay)', color='blue',
             marker='D', linestyle='-')
    plt.plot(network_delay_aseco[:, 0], network_delay_aseco[:, 1], label='NEPTUNE+(Network delay)', color='green',
             marker='s', linestyle='--')
    plt.plot(network_delay_heu_xu[:, 0], network_delay_heu_xu[:, 1], label='HEU(Network delay)', color='black',
             marker='^', linestyle='--')

    if has_decision_delay:
        plt.plot(decision_delay_neptune[:, 0], decision_delay_neptune[:, 1], label='NEPTUNE(Allocation delay)',
                 color='blue',
                 marker='P', linestyle='-')
        plt.plot(decision_delay_aseco[:, 0], decision_delay_aseco[:, 1], label='NEPTUNE+(Allocation delay)',
                 color='green',
                 marker='+', linestyle='--')
        plt.plot(decision_delay_heu_xu[:, 0], decision_delay_heu_xu[:, 1], label='HEU(Allocation delay)', color='black',
                 marker='>', linestyle='--')

    plt.xlabel('Cores on node receiving direct calls (millicores)')
    plt.ylabel('Delay (ms)')
    plt.legend()

    # Customize legend and marker size
    plt.legend(fontsize='small')
    plt.setp(plt.gca().get_legend().get_texts(), fontsize='6')  # Adjust legend font size
    plt.setp(plt.gca().get_legend().get_title(), fontsize='6')  # Adjust legend title font size
    plt.setp(plt.gca().get_lines(), markersize=4)  # Adjust marker size

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    file_name = f'plot_{application}(10nod200req_varyCoresFistNode-delay{cold_start_inclusion}).pdf'
    plt.savefig(os.path.join(save_path, file_name), format='pdf')

    plt.clf()  # clear the current graph


    # +++++++++++++++++++++++++ Memory ++++++++++++++++++++++++++++++++
    plt.figure(figsize=(pdf_sheet_width, pdf_sheet_height))
    # # Plot for matrix A with a specific color, marker, and format
    plt.plot(memory_neptune[:, 0], memory_neptune[:, 1], label='NEPTUNE', color='blue', marker='o', linestyle='-')
    plt.plot(memory_aseco[:, 0], memory_aseco[:, 1], label='NEPTUNE+', color='green', marker='*', linestyle='--')
    plt.plot(memory_heu_xu[:, 0], memory_heu_xu[:, 1], label='HEU', color='black', marker='x', linestyle='--')
    plt.xlabel('Cores on node receiving direct calls (millicores)')
    plt.ylabel('Total memory (MB)')
    plt.legend()
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    file_name = f'plot_{application}(10nod200req_varyCoresFistNode(memory)).pdf'
    plt.savefig(os.path.join(save_path, file_name), format='pdf')

    plt.clf()  # clear the current graph

    # +++++++++++++++++++++++++ TOTAL NODES USED ++++++++++++++++++++++++++++++++
    plt.figure(figsize=(pdf_sheet_width, pdf_sheet_height))
    plt.plot(nodes_neptune[:, 0], nodes_neptune[:, 1], label='NEPTUNE', color='blue', marker='o', linestyle='-')
    plt.plot(nodes_aseco[:, 0], nodes_aseco[:, 1], label='NEPTUNE+', color='green', marker='*', linestyle='--')
    plt.plot(nodes_heu_xu[:, 0], nodes_heu_xu[:, 1], label='HEU', color='black', marker='x', linestyle='--')
    plt.xlabel('Cores on node receiving direct calls (millicores)')
    plt.ylabel('Total nodes used')
    plt.legend()
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    file_name = f'plot_{application}(10nod200req_varyCoresFistNode(total nodes).pdf'

    st_file_name = f'statistic(avg)_{application}' \
                   f'(10nod200req_varyCoresFistNode{cold_start_inclusion}.pdf'

    # plt.savefig(os.path.join(save_path, file_name), format='pdf')

    statistic(cold_start_inclusion, coldstart_delay_aseco, coldstart_delay_heu_xu, coldstart_delay_neptune,
              decision_delay_aseco, decision_delay_heu_xu, decision_delay_neptune, delays_aseco, delays_heu_xu,
              delays_neptune, memory_aseco, memory_heu_xu, memory_neptune, network_delay_aseco, network_delay_heu_xu,
              network_delay_neptune, nodes_aseco, nodes_heu_xu, nodes_neptune, st_file_name)

    mean_delays = st.generate_by_count(st, [delays_neptune, delays_heu_xu, delays_aseco])
    mean_nodes = st.generate_by_count(st, [nodes_neptune, nodes_heu_xu, nodes_aseco])
    mean_memory = st.generate_by_count(st, [memory_neptune, memory_heu_xu, memory_aseco])
    mean_coldstart = st.generate_by_count(st, [coldstart_delay_neptune, coldstart_delay_heu_xu, coldstart_delay_aseco])
    mean_network_delay = st.generate_by_count(st, [network_delay_neptune, network_delay_heu_xu, network_delay_aseco])
    st.create_statistical_table(st, ['Delay', 'Nodes', 'Memory', 'Coldstart', 'Network Delay'], ['NEPTUNE', 'HEU', 'NEPTUNE+'],
                                [mean_delays, mean_nodes, mean_memory, mean_coldstart, mean_network_delay], 'plots', st_file_name)
    plt.clf()  # clear the current graph


    # # NOTE ****** VARY THE NUMBER OF NODES FROM 10 TO 50 ***************
    # lamb = 200  # fix workload for entrypoint function
    #
    # # print("Request received")
    # input = request.json
    #
    # output_matrix_lines = 8
    # topologies = [0, 1, 2, 3, 4, 5, 6, 7]
    # response, sumlamb, delays_neptune, memory_neptune, nodes_neptune, max_delay = \
    #     fill_neptune(output_matrix_lines, workload, input, sumlamb, qty_values, fixed_lamb=lamb, scale=10, max_delay=0,
    #                  topologies=topologies)
    #
    # sumlamb, delays_aseco, memory_aseco, nodes_aseco, max_delay = \
    #     fill_aseco(output_matrix_lines, workload, input, sumlamb, max_delay, fixed_lamb=lamb, scale=10, topologies=topologies)
    #
    # sumlamb, delays_heu_xu, memory_heu_xu, nodes_heu_xu, max_delay = \
    #     fill_heu_xu(output_matrix_lines, workload, input, sumlamb, max_delay, fixed_lamb=lamb, scale=10, topologies=topologies)
    #
    #
    # # Plot for matrix A with a specific color, marker, and format
    # plt.figure(figsize=(pdf_sheet_width, pdf_sheet_height))
    # plt.plot(delays_neptune[:, 0], delays_neptune[:, 1], label='NEPTUNE', color='blue', marker='o', linestyle='-')
    # plt.plot(delays_aseco[:, 0], delays_aseco[:, 1], label='NEPTUNE+', color='green', marker='*', linestyle='--')
    # plt.plot(delays_heu_xu[:, 0], delays_heu_xu[:, 1], label='HEU', color='black', marker='x', linestyle='--')
    # plt.xlabel('Number of nodes')
    # plt.ylabel('Total delay (ms)')
    # plt.legend()
    # if not os.path.exists(save_path):
    #     os.makedirs(save_path)
    # file_name = f'plot_{application}(200req_varyNode_Qty(delay).pdf'
    # plt.savefig(os.path.join(save_path, file_name), format='pdf')
    #
    # plt.clf()  # clear the current graph
    #
    # # +++++++++++++++++++++++++ Memory ++++++++++++++++++++++++++++++++
    # # # Plot for matrix A with a specific color, marker, and format
    # plt.figure(figsize=(pdf_sheet_width, pdf_sheet_height))
    # plt.plot(memory_neptune[:, 0], memory_neptune[:, 1], label='NEPTUNE', color='blue', marker='o', linestyle='-')
    # plt.plot(memory_aseco[:, 0], memory_aseco[:, 1], label='NEPTUNE+', color='green', marker='*', linestyle='--')
    # plt.plot(memory_heu_xu[:, 0], memory_heu_xu[:, 1], label='HEU', color='black', marker='x', linestyle='--')
    # plt.xlabel('Number of nodes')
    # plt.ylabel('Total memory (MB)')
    # plt.legend()
    # if not os.path.exists(save_path):
    #     os.makedirs(save_path)
    # file_name = f'plot_{application}(200req_varyNode_Qty(memory).pdf'
    # plt.savefig(os.path.join(save_path, file_name), format='pdf')
    #
    # plt.clf()  # clear the current graph
    #
    # # +++++++++++++++++++++++++ TOTAL NODES USED ++++++++++++++++++++++++++++++++
    # plt.figure(figsize=(pdf_sheet_width, pdf_sheet_height))
    # plt.plot(nodes_neptune[:, 0], nodes_neptune[:, 1], label='NEPTUNE', color='blue', marker='o', linestyle='-')
    # plt.plot(nodes_aseco[:, 0], nodes_aseco[:, 1], label='NEPTUNE+', color='green', marker='*', linestyle='--')
    # plt.plot(nodes_heu_xu[:, 0], nodes_heu_xu[:, 1], label='HEU', color='black', marker='x', linestyle='--')
    # plt.xlabel('Number of nodes')
    # plt.ylabel('Total nodes used')
    # plt.legend()
    # if not os.path.exists(save_path):
    #     os.makedirs(save_path)
    # file_name = f'plot_{application}(200req_varyNode_Qty(nodes).pdf'
    #
    # st_file_name = f'statistic(avg)_{application}(200req_varyNode_Qty(nodes).pdf'
    #
    # plt.savefig(os.path.join(save_path, file_name), format='pdf')
    #
    # mean_delays = st.generate_by_count(st, [delays_neptune, delays_heu_xu, delays_aseco])
    # mean_nodes = st.generate_by_count(st, [nodes_neptune, nodes_heu_xu, nodes_aseco])
    # mean_memory = st.generate_by_count(st, [memory_neptune, memory_heu_xu, memory_aseco])
    # print(f'mean_delays/mean_nodes/mean_memory ={mean_delays}/{mean_nodes}/{mean_memory}')
    # st.create_statistical_table(st, ['Delay', 'Nodes', 'Memory'], ['NEPTUNE', 'HEU', 'NEPTUNE+'],
    #                             [mean_delays, mean_nodes, mean_memory], 'plots', st_file_name)
    # plt.clf()  # clear the current graph
    return response


def statistic(cold_start_inclusion, coldstart_delay_aseco, coldstart_delay_heu_xu, coldstart_delay_neptune,
              decision_delay_aseco, decision_delay_heu_xu, decision_delay_neptune, delays_aseco, delays_heu_xu,
              delays_neptune, memory_aseco, memory_heu_xu, memory_neptune, network_delay_aseco, network_delay_heu_xu,
              network_delay_neptune, nodes_aseco, nodes_heu_xu, nodes_neptune, st_file_name):
    mean_delays = st.generate_by_count(st, [delays_neptune, delays_heu_xu, delays_aseco])
    mean_allocation_delay = st.generate_by_count(st, [decision_delay_neptune, decision_delay_heu_xu,
                                                      decision_delay_aseco])
    mean_nodes = st.generate_by_count(st, [nodes_neptune, nodes_heu_xu, nodes_aseco])
    mean_memory = st.generate_by_count(st, [memory_neptune, memory_heu_xu, memory_aseco])
    mean_coldstart = st.generate_by_count(st, [coldstart_delay_neptune, coldstart_delay_heu_xu, coldstart_delay_aseco])
    mean_network_delay = st.generate_by_count(st, [network_delay_neptune, network_delay_heu_xu, network_delay_aseco])
    st.create_statistical_table(st, [f'Total delay{cold_start_inclusion}', 'Nodes', 'Memory', 'Coldstart',
                                     'Network delay', 'Allocation delay'], ['NEPTUNE', 'HEU', 'NEPTUNE+'],
                                [mean_delays, mean_nodes, mean_memory, mean_coldstart, mean_network_delay,
                                 mean_allocation_delay], 'plots', st_file_name)


# if __name__ == "__main__":
#     app.run(host='0.0.0.0', port=5000, debug=True)
app.run(host='0.0.0.0', port=5000, threaded=False, processes=10, debug=True)
