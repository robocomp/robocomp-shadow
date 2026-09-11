from pydsr import *
from .mission_base import MissionBase


class GuideMission(MissionBase):
    def __init__(self, graph, LTSM_graph, mission, agent_id):
        super().__init__(graph, LTSM_graph, mission, agent_id)
        # Initial checks
        if not self.missions:
            print("No missions provided. Exiting.")
            return

        # Robot node variables
        robot_nodes = self.graph.get_nodes_by_type("omnirobot") + self.graph.get_nodes_by_type("robot")
        if len(robot_nodes) == 0:
            print("No robot node found in the graph. Exiting")
            exit(0)
        robot_node = robot_nodes[0]

        self.robot_name = robot_node.name
        self.robot_id = robot_node.id

        current_edges = self.graph.get_edges_by_type("current")
        if current_edges:
            actual_room = self.graph.get_node(current_edges[0].destination)
            actual_room_id = actual_room.attrs["room_id"].value
            if actual_room_id is not None:
                self.actual_room_id = actual_room_id

        person_node = self.graph.get_node(self.missions[0]["person"])
        if not person_node:
            print(f"Person node {self.missions[0]['target']} not found in the graph.")
            return

        self.person_id = person_node.attrs["person_id"].value
        target_affordance_node = self.get_current_target_affordance_node(self.missions[0]["person"],
                                                                         "follow")
        self.insert_TODO_edge(target_affordance_node)

        print("Robot node found:", self.robot_name, "with id", self.robot_id)

        print("GuideMission initialized with graph and missions.")

    def monitor(self):
        # Monitor the missions and perform actions
        # 1 - Check the list of missions
        if not self.missions:
            print("No missions to monitor. Mision finished.")
            self.current_bt_state = None
            return True, None
        # 2 - Get the first mission from the list
        mission = self.missions[0]

        if self.current_bt_state == "completed":  # fixed 'is' to '==' for string comparison safety
            print(f"Mission {mission} already completed. Removing it from the list.")
            self.missions.popleft()
            return False, self.missions

        # 3 - Check if the mission is being executed

        # Check actual person current room
        person_node = self.graph.get_node(mission["person"])
        if person_node is None:
            print(f"Person node {mission['person']} not found in the graph.")
            return False, None
        if "room_id" not in person_node.attrs:
            print(f"Person node {mission['person']} does not have a room_id attribute.")
            return False, None
        person_actual_room_id = person_node.attrs["room_id"].value
        if person_actual_room_id is None:
            print(f"Person node {mission['person']} has no room_id value.")
            return False, None

        actual_person_room_node_list = [node.name for node in self.LSTM_graph.get_nodes_by_type("room") if
                                        node.attrs["room_id"].value == person_actual_room_id]
        if not actual_person_room_node_list:
            print(f"No room node found for room_id {person_actual_room_id}.")
            return False, None
        actual_person_room_node = actual_person_room_node_list[0]

        robot_current_room = [edge.destination for edge in self.graph.get_edges_by_type("current")]
        if not robot_current_room:
            print(f"No current room edge found for robot {self.robot_id}.")
            return False, None
        robot_current_room_node = self.graph.get_node(robot_current_room[0])
        if robot_current_room_node is None:
            print(f"Robot current room node {robot_current_room[0]} not found in the graph.")
            return False, None

        if "room_id" in robot_current_room_node.attrs:
            robot_current_room_id = robot_current_room_node.attrs["room_id"].value

            # --- Logic Split: Are they in different rooms or the same room? ---

            if robot_current_room_id != person_actual_room_id:
                # [CASE A] Robot and Person are in DIFFERENT rooms

                if actual_person_room_node == self.target:
                    # If person reached final target, complete mission.
                    print(f"Person {mission['person']} is already in the target room {self.target}. Mission completed.")
                    self.current_bt_state = None
                    target_affordance_node = self.get_current_target_affordance_node(self.missions[0]["person"],
                                                                                     "follow")
                    self.remove_TODO_edge(target_affordance_node)
                    return True, None
                else:
                    # Logic for finding the door and crossing
                    actual_target_node = self.graph.get_node(mission["target"])
                    if actual_target_node is None:
                        print(f"Node {mission['target']} not found.")
                        # self.missions.popleft()
                        return False, None

                    if "connected_room_name" not in actual_target_node.attrs:
                        print(f"Node {mission['target']} does not have connected_room_name attribute.")
                        return False, None

                    actual_target_room = actual_target_node.attrs["connected_room_name"].value

                    if actual_target_room != actual_person_room_node:
                        print(
                            f"Person {mission['person']} is not in the target room {actual_target_room} yet. Continuing mission.")
                        return False, None
                    else:
                        # Person is in the next room, prepare to Cross
                        target_affordance_nodes = [node for node in self.graph.get_nodes_by_type("affordance") if
                                                   node.attrs["parent"].value == actual_target_node.id]
                        if not target_affordance_nodes:
                            print(f"{mission['mission']} affordance node for {mission['target']} not found.")
                            return False, None
                        target_affordance_node = target_affordance_nodes[0]

                        # Check if we are already trying to cross
                        todo_edge = self.graph.get_edge(self.robot_id, target_affordance_node.id, "TODO")
                        if todo_edge is None:
                            # Stop following the person
                            follow_affordance_node = self.get_current_target_affordance_node(self.missions[0]["person"],
                                                                                             "follow")
                            self.remove_TODO_edge(follow_affordance_node)

                            # Start crossing the door
                            target_affordance_node = self.get_current_target_affordance_node(self.missions[0]["target"],
                                                                                             "cross")
                            bt_state_attr = target_affordance_node.attrs["bt_state"].value
                            if bt_state_attr is not None:
                                self.current_bt_state = bt_state_attr
                            print(f"Inserting TODO edge to door cross affordance {target_affordance_node.id}.")
                            TODO_edge = Edge(target_affordance_node.id, self.robot_id, "TODO", self.agent_id)
                            self.graph.insert_or_assign_edge(TODO_edge)

                            return False, None
                        else:
                            # Check the status of the affordance node
                            is_active_attr = target_affordance_node.attrs["active"].value
                            bt_state_attr = target_affordance_node.attrs["bt_state"].value
                            print(f"Monitoring door crossing affordance {target_affordance_node.id}: active={is_active_attr}, bt_state={bt_state_attr}")
                            if bool(is_active_attr) is False and bt_state_attr == "completed":
                                print(f"Door crossing affordance {target_affordance_node.id} completed.")
                                # Remove TODO edge
                                try:
                                    self.graph.delete_edge(self.robot_id, target_affordance_node.id, "TODO")
                                    self.missions.popleft()
                                    return False, self.missions
                                except Exception as e:
                                    print(f"Error deleting TODO edge: {e}")

            else:
                # [CASE B] Robot and Person are in the SAME room
                # This was missing in your original code.

                # Verify we aren't just sitting idle.
                # We need to ensure the robot is "Following" the person.

                # 1. Get the follow affordance
                target_affordance_node = self.get_current_target_affordance_node(self.missions[0]["person"], "follow")

                if target_affordance_node:
                    # 2. Re-insert the TODO edge for following.
                    # Note: insert_TODO_edge in mission_base checks if it already exists, so it's safe to call repeatedly.
                    # print(f"Robot and person in same room ({robot_current_room_id}). Resuming 'follow'.")
                    self.insert_TODO_edge(target_affordance_node)
        return False, None

    def abort(self, set_in_pending=False):
        print("Aborting FollowingMission.")
        # Remove TODO edge if exists
        todo_edge = self.graph.get_edge(self.robot_name, "aff_follow_" + str(self.person_id), "TODO")
        if todo_edge:
            try:
                self.graph.delete_edge(self.robot_name, "aff_follow_" + str(self.person_id), "TODO")
                print("TODO edge deleted.")
            except Exception as e:
                print(f"Error deleting TODO edge: {e}")
        else:
            print("No TODO edge found to delete.")
        self.current_bt_state = None
        if not set_in_pending:
            # Clear missions
            self.missions.clear()
        print("GuideMission aborted and cleared.")