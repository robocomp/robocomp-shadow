from pydsr import *

class InteractMission:
    def __init__(self, graph, mission, agent_id):
        self.robot_resources = ["speaker"]

        self.graph = graph
        self.missions = mission["missions"]
        self.target = mission["target"]
        self.type = mission["mission"]
        self.init_timestamp = mission["timestamp"]
        self.agent_id = agent_id


        self.inner_api = inner_api(self.graph)

        self.current_bt_state = None

        self.crossing_timeout = 5
        self.person_out_max_times = 10
        self.person_out_times = 0

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

        self.act_submission = self.missions[0]

        # Find required door node
        person_node = self.graph.get_node(self.act_submission["target"])
        if person_node is None:
            print(f"Person node {self.act_submission['target']} not found in the graph.")
            return False
        # 4 - Check if the cross affordance node exists
        self.person_id = person_node.attrs["person_id"].value

        print("Robot node found:", self.robot_name, "with id", self.robot_id)
        print("GOTOMission initialized with graph and missions.")

    def monitor(self):
        # Monitor the missions and perform actions
        # 1 - Check the list of missions
        if not self.missions:
            print("No missions to monitor. Mision finished.")
            self.current_bt_state = None
            return True

        # print(f"Monitoring mission: {mission}")
        if self.current_bt_state is "completed":
            print(f"Mission {self.act_submission} already completed. Removing it from the list.")
            self.missions.popleft()
            return False

        actual_affordance_node = self.graph.get_node("aff_"+self.act_submission["mission"]+"_"+str(self.person_id))

        if not actual_affordance_node is None:
            self.monitor_affordance_state(actual_affordance_node)

    # def monitor_following(self):
    #     # Check if

    def monitor_affordance_state(self, affordance_node):
        todo_edge = self.graph.get_edge(self.robot_id, affordance_node.id, "TODO")
        if todo_edge is None:
            bt_state_attr = affordance_node.attrs["bt_state"].value
            if bt_state_attr is not None:
                self.current_bt_state = bt_state_attr
            print(
                f"TODO edge from robot {self.robot_id} to door cross affordance node {affordance_node.id} not found in the graph. Inserting it.")
            TODO_edge = Edge(affordance_node.id, self.robot_id, "TODO", self.agent_id)
            self.graph.insert_or_assign_edge(TODO_edge)
            return False
        else:
            # Check the status of the affordance node
            is_active_attr = affordance_node.attrs["active"].value
            bt_state_attr = affordance_node.attrs["bt_state"].value

            if (is_active_attr is not None) and (bt_state_attr is not None):
                self.current_bt_state = bt_state_attr
                # print("Affordance node activated:", is_active_attr, "with state:", bt_state_attr)
                if (is_active_attr == False) and (bt_state_attr == "completed"):
                    print(f"Mission {affordance_node.name} completed. Removing TODO edge and moving to next mission.")
                    # Remove TODO edge
                    try:
                        self.graph.delete_edge(self.robot_id, affordance_node.id, "TODO")
                    except Exception as e:
                        print(f"Error deleting TODO edge: {e}")
                    # Remove the mission from the list
                    self.missions.pop(0)

    def abort(self, set_in_pending=False):
        print("Aborting FollowingMission.")
        # Remove TODO edge if exists
        todo_edge = self.graph.get_edge(self.robot_name, "aff_"+self.act_submission["mission"]+"_"+str(self.person_id), "TODO")
        if todo_edge:
            try:
                self.graph.delete_edge(self.robot_name, "aff_"+self.act_submission["mission"]+"_"+str(self.person_id), "TODO")
                self.graph.delete_edge("aff_"+self.act_submission["mission"]+"_"+str(self.person_id), self.robot_name,  "TODO")
                print("TODO edge deleted.")
            except Exception as e:
                print(f"Error deleting TODO edge: {e}")
        else:
            print("No TODO edge found to delete.")
        self.current_bt_state = None
        if not set_in_pending:
            # Clear missions
            self.missions.clear()
        print("FollowingMission aborted and cleared.")

    def insert_new_submission(self, submission):
        # Insert submission in front of the list
        self.missions.append(submission)
        self.act_submission = self.missions[0]