import itertools

def total_distance(graph, path):
    distance = 0
    for i in range(len(path) - 1):
        distance += graph[path[i]][path[i+1]]
    return distance

def held_karp_with_exclusions(graph, start, destination_nodes, excluded_nodes):
    n = len(graph)
    nodes = [node for node in destination_nodes if node not in excluded_nodes]
    shortest_distance = float('inf')
    shortest_path = []

    for perm in itertools.permutations(nodes):
        perm = (start,) + perm  # Ensure the start node is always at the beginning
        distance = total_distance(graph, perm)
        if distance < shortest_distance:
            shortest_distance = distance
            shortest_path = perm

    return shortest_path