import matplotlib.pyplot as plt
from long_term_graph import LongTermGraph
from PySide2.QtCore import QPoint


def draw_graph(graph):
    """
    Generates a graph based on a provided adjacency matrix using Kamada-Kawai
    layout algorithm, and adds node names and edges with arrowheads.

    Args:
        graph (AbstractGraph): Used to represent a graph object that contains
            vertices and edges.

    """
    fig1, ax1 = plt.subplots()
    ax1.set_title('LTSM graph')
    fig1.canvas.draw()
    fig1.canvas.flush_events()
    ax1.clear()

    # Obtener las coordenadas de los vértices
    layout = graph.layout("kamada_kawai")  # Utiliza el layout Kamada-Kawai
    # Dibujar los vértices
    x, y = zip(*layout)
    ax1.scatter(x, y, s=100)  # Ajustar el tamaño de los vértices con el parámetro 's'
    # Dibujar las aristas
    for edge in graph.get_edgelist():
        # Print rt_translation attribute
        # Get edge data
        edge_data = graph.get_eid(edge[0], edge[1])
        # print(edge_data["rt"])
        ax1.plot([x[edge[0]], x[edge[1]]], [y[edge[0]], y[edge[1]]], color="grey")
        # add arrow to the edge
        # ax.annotate(edge_data, xy=(x[edge[1]], y[edge[1]]), xytext=(x[edge[0]], y[edge[0]]),
        #                   arrowprops=dict(arrowstyle="->", lw=2))
        ax1.annotate(edge_data, xy=((x[edge[1]] + x[edge[0]]) / 2, (y[edge[1]] + y[edge[0]]) / 2),
                     textcoords="offset points",
                     xytext=(x[edge[0]], y[edge[0]]), ha='center', color='green')

    for i, txt in enumerate([f"Node {i}" for i in range(graph.vcount())]):
        # Get name attribute
        name = graph.vs[i]["name"]
        #name = str(i)
        ax1.annotate(name, (x[i], y[i]), textcoords="offset points", xytext=(0, 10), ha='center')
    # Adapt ax to the graph
    ax1.set_xlim([min(x) - 2, max(x) + 2])
    ax1.set_ylim([min(y) - 2, max(y) + 2])

def find_edge_with_attribute(graph, attribute, value):
    """
    Searches through a graph's edges for an edge with a specific attribute equal
    to a given value. If such an edge is found, it returns it; otherwise, it returns
    `None`.

    Args:
        graph (Graph): Represented as an object that contains a collection of
            edges, where each edge represents a connection between two nodes in
            the graph.
        attribute (attribute): Used to specify the attribute of interest for finding
            an edge in a graph.
        value (object): Used to search for an edge in a graph based on a specific
            attribute.

    Returns:
        edge: An untyped reference to a graph edge that has the specified attribute
        equal to the provided value.

    """
    for edge in graph.es:
        if edge[attribute] == value:
            return edge
    return None

def get_room_edges(graph):
    """
    Iterates over the edges in a graph and adds to an output list any edge connecting
    nodes with "door" in their names.

    Args:
        graph (Graph): Represented as g, which contains a collection of nodes and
            edges that define a graph structure.

    Returns:
        list: A collection of edges from the given graph.

    """
    edges = []
    for edge in g.es:
        source_node = g.vs[edge.source]
        target_node = g.vs[edge.target]
        if "door" in source_node["name"]:
            print("source", source_node["name"], target_node["name"], edge.attributes())
            edges.append(edge)

    return edges

def get_connected_door_nodes(graph, node):
    # Base case: if the node is a door node, return it
    """
    In Java code recursively queries the graph for all nodes connected to a given
    node via doors, returning a list of such nodes.

    Args:
        graph (Graph): Used to represent a graph structure.
        node (GraphNode): Referred to as a node in the graph.

    Returns:
        list: A collection of nodes that are connected to a specific node through
        doors.

    """
    if "door" in node["name"] and node["connected_room_name"] is not None:
        return [node]

    door_nodes = []
    # Recursive case: iterate over the successors of the node
    for successor in graph.successors(node):
        # Get the successor node
        successor_node = graph.vs[successor]
        # Recursively call the function on the successor node
        #print("transform from", node["name"], "to", successor_node["name"])
        door_nodes += get_connected_door_nodes(graph, successor_node)

    return door_nodes

def traverse_graph(graph, current_room, visited=None):
    """
    Navigates through a graph by starting from a given room and visiting all other
    rooms reachable through doors. It keeps track of visited rooms using a list
    and prints information about each room it visits.

    Args:
        graph (Graph): Used to represent a graph with nodes and edges.
        current_room (dict): Represents the current room to be traversed in the graph.
        visited (list): Used to keep track of the rooms that have been visited
            during the traversal process, initialized to an empty list if None.

    Returns:
        list: A collection of strings representing the rooms that have been visited.

    """
    if visited is None:
        visited = list()

    # Add the current room to the visited set
    visited.append(current_room['name'])
    print(f"room: {current_room['name']}")

    # Get the connected door nodes in the current room
    door_nodes = get_connected_door_nodes(graph, current_room)

    # For each door node, get the connected room and recursively traverse the graph from there
    for door_node in door_nodes:
        connected_room_name = door_node['connected_room_name']
        connected_room = graph.vs.find(name_eq=connected_room_name)
        if connected_room['name'] not in visited:
            print(f"door: {door_node['name']}, other side: {door_node['other_side_door_name']}")
            traverse_graph(graph, connected_room, visited)

    return visited



########################
graph = LongTermGraph("graph.pkl")

doors = graph.get_room_objects_transform_matrices_with_name("room_1", "door")
for i in doors:
    print(i[1].t)


g_map = graph.compute_metric_map("room_1")
graph.draw_metric_map(g_map)

r = graph.check_point_in_map(g_map, QPoint(-3000, -4000))
graph.draw_point(QPoint(-3000, -4000))
print("Found in room: ", r)

plt.show()
