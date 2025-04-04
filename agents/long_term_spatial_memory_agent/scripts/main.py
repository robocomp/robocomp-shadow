import matplotlib.pyplot as plt
from long_term_graph import LongTermGraph
from PySide2.QtCore import QPoint


def draw_graph(graph):
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
    for edge in graph.es:
        if edge[attribute] == value:
            return edge
    return None

def get_room_edges(graph):
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
