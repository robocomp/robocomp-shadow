from pydsr import *
from abc import ABC, abstractmethod
from collections import deque
from rapidfuzz import process

class MissionBase(ABC):
    """
    Clase base abstracta para la gestión de misiones de un robot social.
    Proporciona la estructura común y métodos utilitarios.
    """
    def __init__(self, graph, mission, agent_id):
        self.graph = graph
        self.agent_id = agent_id
        self.missions = deque(mission.get("missions", []))
        self.type = mission.get("mission", "")
        self.target = mission.get("target", "")
        self.init_timestamp = mission.get("timestamp", 0)
        self.robot_id = None
        self.robot_name = None
        self.current_bt_state = None

        # Inicializar el robot y sus recursos
        self._initialize_robot()

    def _initialize_robot(self):
        """Busca y configura el nodo del robot en el grafo."""
        robot_nodes = self.graph.get_nodes_by_type("omnirobot") + self.graph.get_nodes_by_type("robot")
        if not robot_nodes:
            print("No robot node found in the graph. Exiting")
            exit(0)
        robot_node = robot_nodes[0]
        self.robot_id = robot_node.id
        self.robot_name = robot_node.name
        print(f"Robot node found: {self.robot_name} with id {self.robot_id}")

    @abstractmethod
    def monitor(self):
        """
        Método abstracto que debe ser implementado por cada clase de misión.
        Contiene la lógica principal para monitorear y gestionar la misión.
        """
        pass

    def get_first_mission_from_queue(self):
        """Retorna la primera misión de la cola sin removerla."""
        return self.missions[0] if self.missions else None

    def insert_TODO_edge(self, affordance_node):
        """Inserta un borde TODO entre el robot y un nodo de affordance."""
        todo_edge = self.graph.get_edge(self.robot_id, affordance_node.id, "TODO")
        if todo_edge is None:
            bt_state_attr = affordance_node.attrs["bt_state"].value
            if bt_state_attr is not None:
                self.current_bt_state = bt_state_attr
            print(f"TODO edge from robot {self.robot_id} to affordance node {affordance_node.id} not found. Inserting it.")
            TODO_edge = Edge(affordance_node.id, self.robot_id, "TODO", self.agent_id)
            self.graph.insert_or_assign_edge(TODO_edge)

    def remove_TODO_edge(self, affordance_node):
        """Remueve un borde TODO entre el robot y un nodo de affordance."""
        try:
            self.graph.delete_edge(self.robot_id, affordance_node.id, "TODO")
            print("TODO edge deleted.")
        except Exception as e:
            print(f"Error deleting TODO edge: {e}")

    def abort_mission(self, clear_missions=False):
        """Aborta la misión actual y limpia la cola de misiones."""
        print("Aborting current mission.")
        todo_edges = self.graph.get_edges_by_type("TODO")
        if todo_edges:
            for edge in todo_edges:
                self.graph.delete_edge(edge.origin, edge.destination, "TODO")
        if clear_missions:
            self.missions.clear()
            print("Mission aborted and queue cleared.")
        self.current_bt_state = None

    def abort_and_store_current_submission(self):
        """Mueve la misión actual al final de la cola y aborta la misión."""
        if self.missions:
            self.missions.rotate(-1)
        self.abort_mission()

    def abort_current_submission(self):
        """elimina del principio de la cola la submisión actual."""
        self.missions.popleft()
        self.abort_mission()
        if self.missions:
            new_submission = self.missions[0]
            new_target_affordance_node = self.get_current_target_affordance_node(new_submission["target"],
                                                                                 new_submission["mission"])
            self.insert_TODO_edge(new_target_affordance_node)

    def insert_new_submission(self, submission):
        """Inserta una nueva submisión al final de la cola."""
        self.missions.appendleft(submission)
        new_submission = self.missions[0]
        new_target_affordance_node = self.get_current_target_affordance_node(new_submission["target"],
                                                                             new_submission["mission"])
        self.insert_TODO_edge(new_target_affordance_node)

    def get_current_target_affordance_node(self, target, mission):
        actual_target_node = self.graph.get_node(target)
        if not actual_target_node:
            print(f"Target node {target} not found in the graph.")
            return None

        affordance_nodes = [node for node in self.graph.get_nodes_by_type("affordance") if
                            node.attrs["parent"].value == actual_target_node.id]
        if not affordance_nodes:
            print(f"{mission} affordance node for {target} not found in the graph.")
            return None

        affordance_names = [node.name for node in affordance_nodes]
        current_affordance_target_name, _, _ = process.extractOne(mission, affordance_names)
        target_affordance_node = self.graph.get_node(current_affordance_target_name)
        if not target_affordance_node:
            print(f"{current_affordance_target_name} affordance node not found in the graph.")
            return None
        print("Current required affordance:", current_affordance_target_name)
        return target_affordance_node


