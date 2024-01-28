# Question 1: Graph Creation and Manipulation

import networkx as nx
import matplotlib.pyplot as plt

def create_graph_adjacency_list(vertices):
    graph = {}
    for vertex in vertices:
        graph[vertex] = []
    return graph

def create_graph_adjacency_matrix(vertices):
    graph = [[0] * len(vertices) for _ in range(len(vertices))]
    return graph

def add_vertex(graph, vertex):
    if vertex not in graph:
        graph[vertex] = []

def add_edge(graph, vertex1, vertex2):
    if vertex2 not in graph[vertex1]:
        graph[vertex1].append(vertex2)
    if vertex1 not in graph[vertex2]:
        graph[vertex2].append(vertex1)

def remove_vertex(graph, vertex):
    if vertex in graph:
        del graph[vertex]
    for vertices in graph.values():
        if vertex in vertices:
            vertices.remove(vertex)

def remove_edge(graph, vertex1, vertex2):
    if vertex2 in graph[vertex1]:
        graph[vertex1].remove(vertex2)
    if vertex1 in graph[vertex2]:
        graph[vertex2].remove(vertex1)

def visualize_graph(graph):
    G = nx.Graph(graph)
    nx.draw(G, with_labels=True, node_size=1000, node_color='skyblue', font_size=12)
    plt.show()


vertices_list = ['A', 'B', 'C', 'D']
graph_adj_list = create_graph_adjacency_list(vertices_list)

add_vertex(graph_adj_list, 'E')
add_edge(graph_adj_list, 'A', 'B')
add_edge(graph_adj_list, 'B', 'C')
add_edge(graph_adj_list, 'C', 'D')
add_edge(graph_adj_list, 'D', 'E')

remove_vertex(graph_adj_list, 'B')
remove_edge(graph_adj_list, 'C', 'D')

visualize_graph(graph_adj_list)

# Question 2: Graph Traversals

from collections import defaultdict, deque

class Graph:
    def __init__(self):
        self.graph = defaultdict(list)

    def add_edge(self, u, v):
        self.graph[u].append(v)

    def dfs_recursive(self, v, visited):
        visited[v] = True
        print(v, end=' ')
        for neighbour in self.graph[v]:
            if not visited[neighbour]:
                self.dfs_recursive(neighbour, visited)

    def dfs_iterative(self, start_vertex):
        visited = [False] * len(self.graph)
        stack = [start_vertex]
        while stack:
            v = stack.pop()
            if not visited[v]:
                print(v, end=' ')
                visited[v] = True
                stack.extend([neighbour for neighbour in self.graph[v] if not visited[neighbour]])

    def bfs(self, start_vertex):
        visited = [False] * len(self.graph)
        queue = deque([start_vertex])
        while queue:
            v = queue.popleft()
            if not visited[v]:
                print(v, end=' ')
                visited[v] = True
                queue.extend([neighbour for neighbour in self.graph[v] if not visited[neighbour]])


graph = Graph()
graph.add_edge(0, 1)
graph.add_edge(0, 2)
graph.add_edge(1, 2)
graph.add_edge(2, 0)
graph.add_edge(2, 3)
graph.add_edge(3, 3)

print("DFS Recursive:")
graph.dfs_recursive(2, [False] * len(graph.graph))

print("\nDFS Iterative:")
graph.dfs_iterative(2)

print("\nBFS:")
graph.bfs(2)

# Question 3: Pathfinding Algorithms

import heapq

class Graph:
    def __init__(self):
        self.graph = {}

    def add_edge(self, u, v, weight):
        if u not in self.graph:
            self.graph[u] = []
        self.graph[u].append((v, weight))

    def dijkstra(self, source):
        distances = {vertex: float('inf') for vertex in self.graph}
        distances[source] = 0
        priority_queue = [(0, source)]

        while priority_queue:
            current_distance, current_vertex = heapq.heappop(priority_queue)

            if current_distance > distances[current_vertex]:
                continue

            for neighbour, weight in self.graph[current_vertex]:
                distance = current_distance + weight
                if distance < distances[neighbour]:
                    distances[neighbour] = distance
                    heapq.heappush(priority_queue, (distance, neighbour))

        return distances

graph = Graph()
graph.add_edge('A', 'B', 3)
graph.add_edge('A', 'C', 2)
graph.add_edge('B', 'C', 1)
graph.add_edge('B', 'D', 6)
graph.add_edge('C', 'D', 5)
graph.add_edge('C', 'E', 4)
graph.add_edge('D', 'E', 2)

source_vertex = 'A'
distances = graph.dijkstra(source_vertex)
print("Shortest distances from vertex", source_vertex + ":", distances)

# Question 4: Graph Properties and Analysis

class Graph:
    def __init__(self):
        self.graph = defaultdict(list)

    def add_edge(self, u, v):
        self.graph[u].append(v)
        self.graph[v].append(u)

    def calculate_degree(self, vertex):
        return len(self.graph[vertex])

    def calculate_density(self):
        num_vertices = len(self.graph)
        num_edges = sum(len(neighbours) for neighbours in self.graph.values()) // 2
        return 2 * num_edges / (num_vertices * (num_vertices - 1))

    def calculate_diameter(self):
        def bfs(source):
            visited = set()
            queue = deque([(source, 0)])
            max_distance = 0
            while queue:
                node, distance = queue.popleft()
                visited.add(node)
                max_distance = max(max_distance, distance)
                for neighbour in self.graph[node]:
                    if neighbour not in visited:
                        queue.append((neighbour, distance + 1))
            return max_distance

        max_diameter = 0
        for vertex in self.graph:
            max_diameter = max(max_diameter, bfs(vertex))
        return max_diameter

graph = Graph()
graph.add_edge(0, 1)
graph.add_edge(0, 2)
graph.add_edge(1, 2)
graph.add_edge(2, 3)
graph.add_edge(3, 4)
graph.add_edge(4, 5)

print("Degree of vertex 2:", graph.calculate_degree(2))
print("Density of the graph:", graph.calculate_density())
print("Diameter of the graph:", graph.calculate_diameter())

# Question 5: Topological Sorting

class Graph:
    def __init__(self):
        self.graph = defaultdict(list)

    def add_edge(self, u, v):
        self.graph[u].append(v)

    def topological_sort(self):
        visited = [False] * len(self.graph)
        stack = []

        def dfs(vertex):
            visited[vertex] = True
            for neighbour in self.graph[vertex]:
                if not visited[neighbour]:
                    dfs(neighbour)
            stack.append(vertex)

        for vertex in self.graph:
            if not visited[vertex]:
                dfs(vertex)

        return stack[::-1]

graph = Graph()
graph.add_edge(5, 2)
graph.add_edge(5, 0)
graph.add_edge(4, 0)
graph.add_edge(4, 1)
graph.add_edge(2, 3)
graph.add_edge(3, 1)

print("Topological sorting order:", graph.topological_sort())

# Question 6: Minimum Spanning Trees

class Graph:
    def __init__(self):
        self.graph = []

    def add_edge(self, u, v, weight):
        self.graph.append((u, v, weight))

    def find_parent(self, parent, vertex):
        if parent[vertex] == -1:
            return vertex
        if parent[vertex] != vertex:
            return self.find_parent(parent, parent[vertex])
        return vertex

    def union(self, parent, rank, u, v):
        u_root = self.find_parent(parent, u)
        v_root = self.find_parent(parent, v)
        if rank[u_root] < rank[v_root]:
            parent[u_root] = v_root
        elif rank[u_root] > rank[v_root]:
            parent[v_root] = u_root
        else:
            parent[v_root] = u_root
            rank[u_root] += 1

    def kruskal(self):
        result = []
        self.graph = sorted(self.graph, key=lambda item: item[2])
        parent = [-1] * len(self.graph)
        rank = [0] * len(self.graph)
        for u, v, weight in self.graph:
            u_root = self.find_parent(parent, u)
            v_root = self.find_parent(parent, v)
            if u_root != v_root:
                result.append((u, v, weight))
                self.union(parent, rank, u_root, v_root)
        return result

graph = Graph()
graph.add_edge(0, 1, 4)
graph.add_edge(0, 7, 8)
graph.add_edge(1, 2, 8)
graph.add_edge(1, 7, 11)
graph.add_edge(2, 3, 7)
graph.add_edge(2, 8, 2)
graph.add_edge(2, 5, 4)
graph.add_edge(3, 4, 9)
graph.add_edge(3, 5, 14)
graph.add_edge(4, 5, 10)
graph.add_edge(5, 6, 2)
graph.add_edge(6, 7, 1)
graph.add_edge(6, 8, 6)
graph.add_edge(7, 8, 7)

print("Minimum Spanning Tree (Kruskal's Algorithm):", graph.kruskal())

# Question 7: Graph Coloring

class Graph:
    def __init__(self):
        self.graph = {}

    def add_edge(self, u, v):
        if u not in self.graph:
            self.graph[u] = []
        self.graph[u].append(v)

    def is_safe(self, vertex, colour, colour_map):
        for neighbour in self.graph[vertex]:
            if colour_map[neighbour] == colour:
                return False
        return True

    def graph_colouring_util(self, num_colours, colour_map, vertex):
        if vertex == len(self.graph):
            return True

        for colour in range(1, num_colours + 1):
            if self.is_safe(vertex, colour, colour_map):
                colour_map[vertex] = colour
                if self.graph_colouring_util(num_colours, colour_map, vertex + 1):
                    return True
                colour_map[vertex] = 0

        return False

    def graph_colouring(self, num_colours):
        colour_map = [0] * len(self.graph)
        if self.graph_colouring_util(num_colours, colour_map, 0):
            return colour_map
        return None

graph = Graph()
graph.add_edge(0, 1)
graph.add_edge(0, 2)
graph.add_edge(1, 2)
graph.add_edge(2, 3)
graph.add_edge(3, 4)

num_colours = 3
colour_map = graph.graph_colouring(num_colours)
if colour_map:
    print("Graph Colouring with", num_colours, "colours:", colour_map)
else:
    print("No feasible graph colouring with", num_colours, "colours.")


# Question 8: A* Search

import heapq

class Graph:
    def __init__(self):
        self.graph = {}

    def add_edge(self, u, v, weight):
        if u not in self.graph:
            self.graph[u] = []
        self.graph[u].append((v, weight))

    def astar(self, start, goal):
        open_set = [(0, start)]
        came_from = {}
        g_score = {node: float("inf") for node in self.graph}
        g_score[start] = 0
        f_score = {node: float("inf") for node in self.graph}
        f_score[start] = self.heuristic(start, goal)

        while open_set:
            current = heapq.heappop(open_set)[1]
            if current == goal:
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start)
                return path[::-1]

            for neighbour, weight in self.graph[current]:
                tentative_g_score = g_score[current] + weight
                if tentative_g_score < g_score[neighbour]:
                    came_from[neighbour] = current
                    g_score[neighbour] = tentative_g_score
                    f_score[neighbour] = g_score[neighbour] + self.heuristic(neighbour, goal)
                    heapq.heappush(open_set, (f_score[neighbour], neighbour))

        return None

    def heuristic(self, node, goal):
        return ((node[0] - goal[0]) ** 2 + (node[1] - goal[1]) ** 2) ** 0.5

graph = Graph()
graph.add_edge((0, 0), (1, 0), 1)
graph.add_edge((1, 0), (1, 1), 1)
graph.add_edge((1, 1), (0, 1), 1)
graph.add_edge((0, 1), (0, 0), 1)

start = (0, 0)
goal = (1, 1)
path = graph.astar(start, goal)
if path:
    print("Shortest path from", start, "to", goal, ":", path)
else:
    print("No path found from", start, "to", goal)

# Question 9: Graph Isomorphism

import networkx as nx

def isomorphic(graph1, graph2):
    return nx.is_isomorphic(graph1, graph2)

graph1 = nx.cycle_graph(4)
graph2 = nx.path_graph(4)

if isomorphic(graph1, graph2):
    print("The two graphs are isomorphic.")
else:
    print("The two graphs are not isomorphic.")

# Question 10: Graph Optimization

class Graph:
    def __init__(self):
        self.graph = {}

    def add_edge(self, u, v, weight):
        if u not in self.graph:
            self.graph[u] = {}
        self.graph[u][v] = weight

graph = Graph()
graph.add_edge('A', 'B', 10)
graph.add_edge('A', 'C', 15)
graph.add_edge('A', 'D', 20)
graph.add_edge('B', 'C', 35)
graph.add_edge('B', 'D', 25)
graph.add_edge('C', 'D', 30)



