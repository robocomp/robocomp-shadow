from pydsr import *

class GuideMission:
    def __init__(self, graph, mission, agent_id):
        self.graph = graph
        self.missions = mission["missions"]
        self.target = mission["target"]
        self.type = mission["mission"]
        self.init_timestamp = mission["timestamp"]
        self.agent_id = agent_id

        self.current_bt_state = None

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

        self.person_id = None

        print("Robot node found:", self.robot_name, "with id", self.robot_id)

        print("GuideMission initialized with graph and missions.")


    def monitor(self):
        # Monitor the missions and perform actions
        # 1 - Check the list of missions
        if not self.missions:
            print("No missions to monitor. Mision finished.")
            self.current_bt_state = None
            return True
        # 2 - Get the first mission from the list
        mission = self.missions[0]
        # print(f"Monitoring mission: {mission}")
        if self.current_bt_state is "completed":
            print(f"Mission {mission} already completed. Removing it from the list.")
            self.missions.popleft()
            return False
        # 3 - Check if the mission is beign executed checking
        # Find required door node
        person_node = self.graph.get_node(mission[1])
        if person_node is None:
            print(f"Person node {mission[1]} not found in the graph.")
            return False
        # 4 - Check if the cross affordance node exists
        self.person_id = person_node.attrs["person_id"].value
        if self.person_id:
            person_following_affordance_node = self.graph.get_node("aff_"+mission[0]+"_"+str(self.person_id))
            if person_following_affordance_node is None:
                print(f"Cross affordance node aff_{mission[0]}_{str(self.person_id)} not found in the graph.")
                return False
            # 5 Check if TODO edge exists from the robot to the door cross affordance node
            todo_edge = self.graph.get_edge(self.robot_id, person_following_affordance_node.id, "TODO")
            if todo_edge is None:
                bt_state_attr = person_following_affordance_node.attrs["bt_state"].value
                if bt_state_attr is not None:
                    self.current_bt_state = bt_state_attr
                print(f"TODO edge from robot {self.robot_id} to door cross affordance node {person_following_affordance_node.id} not found in the graph. Inserting it.")
                TODO_edge = Edge(person_following_affordance_node.id, self.robot_id,  "TODO", self.agent_id)
                self.graph.insert_or_assign_edge(TODO_edge)
                return False
            else:
                # Check the status of the affordance node
                is_active_attr = person_following_affordance_node.attrs["active"].value
                bt_state_attr = person_following_affordance_node.attrs["bt_state"].value

                if (is_active_attr is not None) and (bt_state_attr is not None):
                    self.current_bt_state = bt_state_attr
                    # print("Affordance node activated:", is_active_attr, "with state:", bt_state_attr)
                    if (is_active_attr == False) and (bt_state_attr == "completed"):
                        print(f"Mission {mission} completed. Removing TODO edge and moving to next mission.")
                        # Remove TODO edge
                        try:
                            self.graph.delete_edge(todo_edge)
                        except Exception as e:
                            print(f"Error deleting TODO edge: {e}")
                        # Remove the mission from the list
                        self.missions.pop(0)

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
        print("GOTOMission aborted and cleared.")