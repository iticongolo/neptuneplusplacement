import copy
from collections import defaultdict
import re

import networkx as nx
import numpy as np
from matplotlib import pyplot as plt



def DFSUtil(node, graph, visited, path, paths):
    visited[node] = True
    path.append(node)
    if not graph[node]:
        paths.append(path.copy())
    for neighbor in graph[node]:
        if not visited[neighbor]:
            DFSUtil(neighbor, graph, visited, path, paths)
    path.pop()
    visited[node] = False


def getAllFullPaths(graph, start):
    visited = defaultdict(bool)
    paths = []
    DFSUtil(start, graph, visited, [], paths)
    return paths

def to_mil(number):
    return number*1000

def inv_mil(number):
    return number/1000


class DAG:
    def __init__(self, dag, startNodename):
        self.dag = dag
        self.startNodeName = startNodename
        nx.set_node_attributes(dag, dag.nodes, 'users')
        nx.set_node_attributes(dag, dag.nodes, 'node')
        nx.set_edge_attributes(dag, dag.edges, 'sync')  # synchrone or asynchrone calls
        nx.set_edge_attributes(dag, dag.edges, 'times')  # for cycles
        nx.set_node_attributes(dag, dag.nodes, 'app')

        nx.set_node_attributes(dag, dag.nodes, 'total_node_req')
        nx.set_node_attributes(dag, dag.nodes, 'rt')
        nx.set_node_attributes(dag, dag.nodes, 'st')
        nx.set_node_attributes(dag, dag.nodes, 'lrt')
        nx.set_node_attributes(dag, dag.nodes, 'cores')
        nx.set_node_attributes(dag, dag.nodes, 'cores_deviation')
        nx.set_node_attributes(dag, dag.nodes, 'rt_deviation')
        nx.set_node_attributes(dag, dag.nodes, 'lrt_deviation')

        self.maxTotalRTs = {}
        self.minTotalRTs = {}
        self.AvgTotalRTs = {}
        self.totalRts = {}
        self.deviationTotalRTs = {}

        self.maxCores = {}
        self.minCores = {}
        self.avgCores = {}
        self.cores = {}
        self.deviationCores = {}

        self.maxLocalRTs = {}
        self.minLocalRTs = {}
        self.AvgLocalRTs = {}
        self.localRts = {}
        self.deviationLocalRTs = {}

        self.maxMinNotIntialized = True
        self.contInterestingRTsCores = 0

    # Join multiple lists of node names (Strings) into one without repetitions
    def uniqueList(self, lists):
        listNames = []
        for list in lists:
            for i in range(0, len(list)):
                nodeName = list[i]
                if nodeName not in listNames:
                    listNames.append(nodeName)
        return listNames

    # def get_predecessors(self, node, visited=None):
    #     if visited is None:
    #         visited = set()
    #
    #     predecessors = set()
    #
    #     def dfs(current_node):
    #         visited.add(current_node)
    #         for neighbor in self.graph[current_node]:
    #             if neighbor not in visited:
    #                 predecessors.add(neighbor)
    #                 dfs(neighbor)
    #
    #     dfs(node)
    #     return predecessors
    def ancestors(self, target_node):
        return list(nx.topological_sort(self.dag.subgraph(nx.ancestors(self.dag, target_node))))

    def getNode(self, nodeName):

        return self.dag.nodes[nodeName]['node']

    def sub_dag(self, startNodeName):
        subdag = {}
        visited = {startNodeName}
        queue = [startNodeName]

        while queue:
            node = queue.pop(0)
            for neighbor in self.dag[node]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
                    subdag.setdefault(node, set()).add(neighbor)
        return subdag

    # convert a Dag into a list given a start point -- DONE
    def toList(self, startNode):
        lists = getAllFullPaths(self.dag, startNode)
        newList = []
        listNames = []
        for nodeNamelist in lists:
            nodeName = nodeNamelist[0]
            node =self.dag.nodes[nodeName]['node']
            nodeNamelist.pop(0)
            if nodeName not in listNames:
                newList.append(node)
                listNames.append(nodeName)
            for nodeName in nodeNamelist:
                if nodeName not in listNames:
                    node = self.dag.nodes[nodeName]['node']
                    newList.append(node)
                    listNames.append(nodeName)
        return newList

    # sum and return the RT of given Node (given by name) considering dependencies
    def getNodeRT(self, nodeName):
        rt = self.getNode(nodeName).app.RT
        return rt

    def getNodeWeight(self, nodeName):
        weight = self.getNode(nodeName).subtotal_weight
        if weight == 0:
            weight = self.getNode(nodeName).app.weight
        return weight

    # called after users updated, only for cycles
    def updateUsersListForCycles(self, father, children, t):
        if len(children) == 0:
            return
        for child in children:
            node = self.dag.nodes[child]['node']
            app, gen = node.app, node.generator
            childNewusers = int(gen.tick(t))
            times = self.getEdgeValue(father, node.name, 'times')
            # we consider the last users generated in father and local child new users
            ti = 1 if (times < 1 or None) else times
            total_new_users = app.users + self.getNode(father).app.users * ti + childNewusers
            app.users = total_new_users
            new_rt = node.app.getRT(total_new_users)
            app.RT = new_rt  # Set Local RT
            node.total_rt = app.RT
            childrenList = self.get_children(child)
            self.updateUsersListForCycles(child, childrenList, t)

    def getEdgeValue(self, node1, node2, attribute):
        return self.dag.edges[node1, node2][attribute]

    def resetUsers(self, start):
        dagList = self.toList(start)
        for node in dagList:
            self.dag.nodes[node.name]['node'].app.users = 0

    def get_predecessors(self, f):
        return list(self.dag.predecessors(f))

    # given a sorted list of nodes return the list of indexes containing predecessors of f
    def get_predecessors_indexes(self, f):
        list_dag = sorted(list(self.dag.nodes))
        # print(f'***********************************************************************************sorted(list(self.dag.nodes)={list_dag}')
        predecessor = self.get_predecessors(f)
        predecessor_indexes = []
        for value in predecessor:
            predecessor_indexes.append(list_dag.index(value))
        return predecessor_indexes

    def get_successors(self, f):
        return list(self.dag.successors(f))

    def __get_successors_edge_ids(self, f, sequential=True):
        successors = self.get_successors(f)
        parallel_ids = []
        sequential_ids = []
        for successor in successors:
            # idf = self.getEdgeValue('f0', successor, 'sync')
            idf = self.dag.edges[(f, successor)]['sync']
            if idf in sequential_ids:
                sequential_ids.remove(idf)
                if idf not in parallel_ids:
                    parallel_ids.append(idf)
            else:
                if idf not in parallel_ids:
                    sequential_ids.append(idf)
        if sequential:
            return sequential_ids
        return parallel_ids

    def get_sequential_successors(self, f):
        edge_ids = self.__get_successors_edge_ids(f, sequential=True)
        successors = self.get_successors(f)
        sequential_successors = []
        for successor in successors:
            idf = self.dag[f][successor].get('sync')
            if idf in edge_ids:
                sequential_successors.append(successor)
        return sequential_successors

    # def get_parallel_successors(self, f):
    #     edge_ids = self.__get_successors_edge_ids(f, sequential=False)
    #     successors = self.get_successors(f)
    #     parallel_successors = []
    #     for successor in successors:
    #         idf = self.dag[f][successor]['sync']
    #         if idf in edge_ids:
    #             parallel_successors.append(successor)
    #     return parallel_successors

    def get_parallel_successors(self, f):
        edge_ids = self.__get_successors_edge_ids(f, sequential=False)
        successors = self.get_successors(f)

        # Create a dictionary to group nodes by their 'sync' values
        sync_groups = {}

        for successor in successors:
            sync_value = self.dag[f][successor].get('sync')  # Assuming 'sync' is an attribute for the edges
            if sync_value in edge_ids:
                if sync_value not in sync_groups:
                    sync_groups[sync_value] = []
                sync_groups[sync_value].append(successor)

        # Convert the dictionary values to a list of lists
        parallel_successors = list(sync_groups.values())

        return parallel_successors

    # given a sorted list of nodes return the list of indexes containing sequential successors of f
    def get_sequential_successors_indexes(self, f):

        function_names = list(self.dag.nodes)

        def custom_sort(name):
            # Extract the numerical part using regular expression
            num_part = re.search(r'\d+', name).group()
            # Convert numerical part to integer
            num = int(num_part)
            return num
        # Sort the list using the custom sorting function
        sorted_function_names = sorted(function_names, key=custom_sort)
        list_dag = sorted_function_names
        successors = self.get_sequential_successors(f)
        successors_indexes = []
        for value in successors:
            successors_indexes.append(list_dag.index(value))
        return successors_indexes

    # given a sorted list of nodes return the list of indexes containing the parallel successors of f
    def get_parallel_successors_indexes(self, f):
        function_names = list(self.dag.nodes)

        # def custom_sort(name):
        #     # Split the string into prefix and numerical part
        #     prefix, num_part = name[0], name[1:]
        #     # Convert numerical part to integer
        #     num = int(num_part)
        #     return (prefix, num)
        def custom_sort(name):
            # Extract the numerical part using regular expression
            num_part = re.search(r'\d+', name).group()
            # Convert numerical part to integer
            num = int(num_part)
            return num

        # Sort the list using the custom sorting function
        sorted_function_names = sorted(function_names, key=custom_sort)
        list_dag = sorted_function_names
        successors = self.get_parallel_successors(f)
        successors_indexes = []
        for i in range(len(successors)):
            parallel_successor_indexes = []
            for j in range(len(successors[i])):
                parallel_successor_indexes.append(list_dag.index(successors[i][j]))
            successors_indexes.append(parallel_successor_indexes)

        return successors_indexes

    def resetweights(self, start):
        dagList = self.toList(start)
        for node in dagList:
            self.dag.nodes[node.name]['node'].subtotal_weight=self.dag.nodes[node.name]['node'].app.weight
    # Only simple DAG was considered and set Local (simple) RT
    def updateUsersDAG(self, startNodeName, t): # start is
        node = self.dag.nodes[startNodeName]['node']
        app, gen = node.app, node.generator
        # currentusersapp=app.numberusers # first is zero
        newusers = int(gen.tick(t))
        users_app = app.users + newusers
        app.users = users_app
        app.setRT(users_app)  # Set Local RT
        node.total_rt = app.RT
        childrenList = self.get_children(startNodeName)
        # self.updateUsersList(childrenList, newusers, t)
        self.updateUsersListForCycles(startNodeName, self.get_children(startNodeName), t)

    def setCores(self, startNodeName, t):
        listDAG=self.toList(startNodeName)
        for node in listDAG:
            app = node.app
            total_rt = self.getNodeRT(node.name)
            mo = node.monitoring
            cont = node.controller
            mo.tick(t, app.RT, total_rt, app.users, app.cores)  # TODO: add local RT
            cores_app = cont.tick(t)
            app.cores = cores_app

    def getControllers(self):
        controllers=[]
        listNodes=self.toList(self.startNodeName)
        for node in listNodes:
            controllers.append(node.controller)
        return controllers

    def getGenerators(self):
        generators  = []
        listNodes = self.toList(self.startNodeName)
        for node in listNodes:
            generators .append(node.generator)
        return generators

    def getMonitorings(self):
        monitorings = []
        listNodes = self.toList(self.startNodeName)
        for node in listNodes:
            monitorings.append(node.monitoring)
        return monitorings

    def get_children(self, node):
        children = []
        edges = self.dag.out_edges(node, data=True)
        for edge in edges:
            children.append(edge[1])
        return children

    def get_nodes_with_children_no_grandChildren(self, dag):
        nodes_with_no_grand_children = set()
        for node in dag.nodes():
            if dag.out_degree(node) == 0:
                continue
            all_children_have_no_children = True
            for child in dag.successors(node):
                if dag.out_degree(child) > 0:
                    all_children_have_no_children = False
                    break
            if all_children_have_no_children:
                nodes_with_no_grand_children.add(node)
        return nodes_with_no_grand_children

    def getTotalNodeRT(self,  rootNodeName):
        totalRT=0
        #for nodeName in list:
        exec_order = self.get_unique_edge_values(rootNodeName)
        for edge_value in exec_order:
            max_rt_node = self.get_max_rt_child_node(self.dag, rootNodeName, edge_value)
            totalRT += max_rt_node.total_rt  # sum the RT of children, use only the max for async nodes
        totalRT += self.getNodeRT(rootNodeName)
        if rootNodeName == 'f0':
            print(rootNodeName, '.Local=', self.getNode(rootNodeName).app.RT,'/',self.getNode(rootNodeName).total_rt )
        return totalRT

    def getTotalNodeWeight(self,  rootNodeName):
        #list=self.get_children(rootNodeName)
        totalWeight=0
        #for nodeName in list:
        exec_order = self.get_unique_edge_values(rootNodeName)
        for edge_value in exec_order:
            max_rt_node = self.get_max_rt_child_node(self.dag, rootNodeName, edge_value)
            totalWeight += max_rt_node.subtotal_weight  # sum the Weight of children, use only the max for parallel nodes
        totalWeight += self.getNodeWeight(rootNodeName)
        return totalWeight

    def setAllRT(self):
        cloned_dag = copy.deepcopy(self.dag)
        visited_nodes = []
        while len(cloned_dag) > 1:
            nodes_names_without_grand_child = self.get_nodes_with_children_no_grandChildren(cloned_dag)
            for nodeName in nodes_names_without_grand_child:
                new_rt = self.getTotalNodeRT(nodeName)
                self.getNode(nodeName).total_rt = new_rt  # set RT to our MAP
                cloned_dag.nodes[nodeName]['node'].total_rt = new_rt  # set RT to the cloned MAP
                children = self.get_children(nodeName)
                visited_nodes.append(children)
            unique_list = self.uniqueList(visited_nodes)
            cloned_dag.remove_nodes_from(unique_list)
    def setAllWeights(self):
        cloned_dag = copy.deepcopy(self.dag)
        visited_nodes = []
        while len(cloned_dag) > 1:
            nodes_names_without_grand_child = self.get_nodes_with_children_no_grandChildren(cloned_dag)
            for nodeName in nodes_names_without_grand_child:
                new_weight = self.getTotalNodeWeight(nodeName)
                # print('RT-Local=', self.getNode(nodeName).app.RT)
                # print('RT=', new_rt)
                self.getNode(nodeName).subtotal_weight = new_weight  # set Global RT (Total weight for a Node) to our DAG
                cloned_dag.nodes[nodeName]['node'].subtotal_weight = new_weight   # set Global RT (Total weight for a Node) to our cloned DAG
                children = self.get_children(nodeName)
                visited_nodes.append(children)
            unique_list = self.uniqueList(visited_nodes)# TODO consider just using children
            cloned_dag.remove_nodes_from(unique_list)

    def setST(self, alfa):
        self.setAllWeights()
        for node in self.dag:
            new_node = self.dag.nodes[node]['node']
            # newst= new_node.app.sla*alfa*new_node.subtotal_weight/new_node.total_weight
            newst = alfa * new_node.app.weight / new_node.total_weight
            new_node.controller.setST(newst)

    def get_children_nodes(self, start_node, edge_value):
        children_nodes = []
        # Check all outgoing edges of the start node
        for child, edge in self.dag[start_node].items():
            new_edge = {key: value for key, value in edge.items() if key == 'sync'}
            if new_edge['sync'] == edge_value:
                children_nodes.append(child)
        return children_nodes

    def get_max_rt_child_node(self, dag, start_nodeName, edge_value):
        children_nodes_names = self.get_children_nodes(start_nodeName, edge_value)
        max_rt_child_node_name = children_nodes_names[0]
        max_nod = self.getNode(max_rt_child_node_name)
        # Check all outgoing edges of the start node
        for child_name, edge in dag[start_nodeName].items():
            new_edge = {key: value for key, value in edge.items() if key == 'sync'}
            # child = self.getNode(child_name)
            if new_edge == edge_value:
                child = self.getNode(child_name)
                if child.total_rt > max_nod.total_rt:
                    max_nod = child
        return max_nod

    def get_unique_edge_values(self, nodeName):
        aux_list = []
        edge_values = []
        for key, value in self.dag[nodeName].items():
            if 'sync' in value:
                # Add value to the array for key='sync'
                aux_list.append(value['sync'])
        # Check all outgoing edges of the node
        for sync_edge in aux_list:
            if sync_edge not in edge_values:
                edge_values.append(sync_edge)
        return edge_values

    # VISUALIZATION
    def __initializeRTs(self):
            for node in self.toList(self.startNodeName):
                rt = node.total_rt
                lrt = node.app.RT
                self.maxTotalRTs.update({node.name: rt})
                self.minTotalRTs.update({node.name: rt})
                self.AvgTotalRTs.update({node.name: rt})
                self.totalRts.update({node.name: [rt]})

                self.maxLocalRTs.update({node.name: lrt})
                self.minLocalRTs.update({node.name: lrt})
                self.AvgLocalRTs.update({node.name: lrt})
                self.localRts.update({node.name: [lrt]})
            self.contInterestingRTsCores = 1
    def __initializeCores(self):
        for node in self.toList(self.startNodeName):
            cores = node.app.cores
            self.maxCores.update({node.name: cores})
            self.minCores.update({node.name: cores})
            self.avgCores.update({node.name: cores})
            self.cores.update({node.name: [cores]})
            self.contInterestingRTsCores = 1

    def to_mil(self, dictionary):
        for key in dictionary.keys():
            dictionary.update({key: dictionary[key] * 1000})

    #Local
    def fillStatiscal(self, starpointUsers = 1):
        for node in self.toList(self.startNodeName):
            users = node.app.users
            if users < starpointUsers:
                break
            else:
                if self.maxMinNotIntialized:
                    self.__initializeCores()
                    self.__initializeRTs()
                    self.maxMinNotIntialized = False
                    return
                else:
                    nodeName = node.name
                    rt = node.total_rt
                    lrt = node.app.RT
                    cores = node.app.cores
                    self.totalRts[nodeName].append(rt)
                    self.localRts[nodeName].append(lrt)
                    self.cores[nodeName].append(cores)

                    if self.maxTotalRTs[nodeName] < rt:
                        self.maxTotalRTs.update({nodeName: rt})

                    if self.minTotalRTs[nodeName] > rt:
                        self.minTotalRTs.update({nodeName: rt})

                    if self.maxLocalRTs[nodeName] < lrt:
                        self.maxLocalRTs.update({nodeName: lrt})

                    if self.minLocalRTs[nodeName] > lrt:
                        self.minLocalRTs.update({nodeName: lrt})

                    if self.maxCores[nodeName] < cores:
                        self.maxCores.update({nodeName: cores})

                    if self.minCores[nodeName] > cores:
                        self.minCores.update({nodeName: cores})
        # print('SUM RT=', self.sumRTs)
        # print('SUM CORES=', self.sumCores)
        self.contInterestingRTsCores += 1

    # Public
    def setStatistical(self):
        self.to_mil(self.minTotalRTs)
        self.to_mil(self.maxTotalRTs)
        self.to_mil(self.minLocalRTs)
        self.to_mil(self.maxLocalRTs)

        self.to_mil(self.minCores)
        self.to_mil(self.maxCores)

        for nodeName in self.totalRts.keys():
            self.AvgTotalRTs.update({nodeName: round(to_mil(np.mean(self.totalRts[nodeName])), 2)})
            self.AvgLocalRTs.update({nodeName: round(to_mil(np.mean(self.localRts[nodeName])), 2)})
            self.avgCores.update({nodeName: round(to_mil(np.mean(self.cores[nodeName])), 2)})
            self.deviationTotalRTs.update({nodeName: round(to_mil(np.std(self.totalRts[nodeName])), 2)})
            self.deviationLocalRTs.update({nodeName: round(to_mil(np.std(self.localRts[nodeName])), 2)})
            self.deviationCores.update({nodeName: round(to_mil(np.std(self.cores[nodeName])), 2)})

    # For DAG visualization only
    def updateForVisualization(self, start):  # start is the root of the MAP
        nodeNameList=self.uniqueList(getAllFullPaths(self.dag, start))
        for nodeName in nodeNameList:
            self.dag.nodes[nodeName]['users'] = self.dag.nodes[nodeName]['node'].app.users
            self.dag.nodes[nodeName]['st'] = round(to_mil(self.dag.nodes[nodeName]['node'].controller.st), 2)
            self.dag.nodes[nodeName]['rt'] = self.AvgTotalRTs[nodeName]     # get from dictionary
            self.dag.nodes[nodeName]['lrt'] = self.AvgLocalRTs[nodeName]  # get from dictionary
            self.dag.nodes[nodeName]['rt_deviation'] = self.deviationTotalRTs[nodeName]  # get from dictionary
            self.dag.nodes[nodeName]['lrt_deviation'] = self.deviationLocalRTs[nodeName]  # get from dictionary
            self.dag.nodes[nodeName]['cores'] = self.avgCores[nodeName]     # get from dictionary
            self.dag.nodes[nodeName]['cores_deviation'] = self.deviationCores[nodeName]  # get from dictionary
            self.dag.nodes[nodeName]['app'] = 'f'

    def print_dag(self, node_content=None, sync='sync', times='times', show_sync=True, show_times=True, seed_value=523):
      #for seed_value in [40,12,5,63,177,377,426,523]: #523 good range(540,600)
             # 40,12,5,63,177,377,426,523]
            # seed_value = 45#40 #12 #5 63 177

            # nx.nx_agraph.pygraphviz_layout(self.dag, prog='/usr/local/bin/dot')
            # output_file = "dag1.pdf"
            # plt.savefig(output_file, format="pdf")

            pos = nx.spring_layout(self.dag, seed=seed_value, scale=2)
            # pos = nx.spring_layout(self.dag, scale=2)
            #nx.kamada_kawai_layout(self.dag)
            # Set figure size and margins
            fig, ax = plt.subplots(figsize=(12, 10))
            fig.subplots_adjust(left=0.001, right=1, top=1, bottom=0.001)
            node_labels = {node: node for node in self.dag.nodes}
            # Draw nodes with labels and attribute values
            if node_content is not None:
                node_labels = {node: f"{node} ({attrs[node_content]})" for node, attrs in self.dag.nodes(data=True)}
            nx.draw_networkx_labels(self.dag, pos, labels=node_labels, font_size=12, font_weight='700',
                                        font_color='darkblue')
            nx.draw_networkx_nodes(self.dag, pos, node_color='none', edgecolors='darkblue', node_size=5000)


            # Draw edges with labels and attribute values
            edge_labels_sync_decorated = nx.get_edge_attributes(self.dag, sync)
            edge_labels_times_decorated = nx.get_edge_attributes(self.dag, times)
            edge_labels_sync = nx.get_edge_attributes(self.dag, sync)  # 'sync'
            edge_labels = copy.deepcopy(edge_labels_sync)
            edge_labels_times = nx.get_edge_attributes(self.dag, times)

            for key in edge_labels_sync:
                edge_labels_sync_decorated[key] = f'({"s"}{edge_labels_sync[key]})'
                edge_labels_times_decorated[key] = f'({"t"}{edge_labels_times[key]})'
                edge_labels[key] = f'({"s"}{edge_labels_sync[key]},{"t"}{edge_labels_times[key]})'
            nx.draw_networkx_edges(self.dag, pos, width=2, alpha=1, edge_color='darkblue', connectionstyle='arc3,rad=0.165')
            if show_sync and show_times:
                nx.draw_networkx_edge_labels(self.dag, pos, edge_labels=edge_labels, font_size=12, font_color='darkblue')
            else:
                if show_sync:
                    nx.draw_networkx_edge_labels(self.dag, pos, edge_labels=edge_labels_sync_decorated, font_size=12, font_color='darkblue')
                else:
                    if show_times:
                        nx.draw_networkx_edge_labels(self.dag, pos, edge_labels=edge_labels_times_decorated, font_size=12,
                                                     font_color='darkblue')
            ax.axis('off')
            font1 = {'family': 'serif', 'color': 'darkblue', 'size': 15, 'weight': 700}
            plt.title(node_content, fontdict=font1, loc='left')
            plt.title(seed_value, fontdict=font1, loc='left')

            self.selectNode("f0")
            if node_content is None:
                node_content = "simple"
            plt.savefig("dag-%s.pdf" % (node_content))
            plt.show()
            plt.close()

    def selectNode(self, node,  seed_value = 523):
        pos = nx.spring_layout(self.dag, seed=seed_value, scale=2)
        edge_width_outer = 100  # Thickness of outer border
        edge_width_inner = 1  # Thickness of inner border

        # Draw outer border for "F1" node
        nx.draw_networkx_nodes(self.dag, pos, nodelist=[node], node_color='white', edgecolors="black", linewidths=edge_width_outer)
        # Draw inner border for "F1" node
        #nx.draw_networkx_nodes(self.dag, pos, nodelist=[node],node_color='white', edgecolors='red', linewidths=edge_width_inner)