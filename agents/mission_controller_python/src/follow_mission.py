from pydsr import *

class FollowMission:
    def __init__(self, graph, mission, agent_id):
        self.robot_resources = ["base", "speaker"]

        self.graph = graph
        self.missions = mission["missions"]
        self.target = mission["target"]
        self.type = mission["mission"]
        self.init_timestamp = mission["timestamp"]
        self.agent_id = agent_id


        self.current_mission = self.missions[0] if self.missions else None

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

        # Find required door node
        person_node = self.graph.get_node(self.current_mission["target"])
        if person_node is None:
            print(f"Person node {self.current_mission['target']} not found in the graph.")
            return False
        # 4 - Check if the cross affordance node exists
        self.person_id = person_node.attrs["person_id"].value

        print("Robot node found:", self.robot_name, "with id", self.robot_id)
        print("GOTOMission initialized with graph and missions.")

    def monitor(self):
        self.current_mission = self.missions[0] if self.missions else None
        # Monitor the missions and perform actions
        # 1 - Check the list of missions
        if not self.missions:
            print("No missions to monitor. Mision finished.")
            self.current_bt_state = None
            return True
        # # 2 - Get the first mission from the list
        # print(f"Monitoring mission: {mission}")
        if self.current_bt_state is "completed":
            print(f"Mission {self.current_mission['mission']} already completed. Removing it from the list.")
            self.missions.popleft()
            self.current_mission = self.missions[0] if self.missions else None
            return False

        actual_target_node = None
        # 3 - Check sub-mission type
        # Find required door node
        actual_target_node = self.graph.get_node(self.current_mission["target"])
        if actual_target_node is None:
            print(f"Target node {self.current_mission['target']} not found in the graph.")
            return False
        # 4 - Check if the cross affordance node exists
        actual_target_nodes = [node for node in self.graph.get_nodes_by_type("affordance") if
                                   node.attrs["parent"].value == actual_target_node.id]
        if not actual_target_nodes:
            print(
                f"{self.current_mission['mission']} affordance node for {self.current_mission['target']} not found in the graph.")
            return False
        target_affordance_node = actual_target_nodes[0]
        match self.current_mission['mission']:
            case "cross":
                pass
            case "follow":
                print("Current missions:", self.missions)
                # Find required door node
                person_node = self.graph.get_node(self.current_mission['target'])
                if person_node is None:
                    print(f"Person node {self.current_mission['target']} not found in the graph.")
                    return False
                # Check if any other missions are pending
                if len(self.missions) > 1:
                    # Get the next mission
                    next_mission = self.missions[1]
                    # Check if the next mission is a cross mission
                    if next_mission["mission"] == "cross":
                        robot_node = self.graph.get_node(self.robot_id)
                        act_time = robot_node.attrs["timestamp_alivetime"].value

                        # Find required door node
                        actual_door_node = self.graph.get_node(next_mission['target'])
                        if actual_door_node is None:
                            print(f"Door node {next_mission['target']} not found in the graph.")
                            return False

                        # Check person pose respect to the door
                        person_pose_respect_to_door = self.inner_api.get_translation_vector(next_mission['target'], self.current_mission['target'])
                        # print(f"Person pose respect to door {next_mission[1]}_{next_mission[2]}: {person_pose_respect_to_door}")
                        if person_pose_respect_to_door[1] > 100:
                            self.person_out_times += 1
                            if self.person_out_times >= self.person_out_max_times:
                                print("Person enough times out of the door. Changing mission to cross affordance.")
                                self.missions.append(self.missions[0])
                                self.missions.pop(0)
                                self.person_out_times = 0
                                self.abort(True)
                                return False
                        else:
                            self.person_out_times = 0
                            print(act_time, next_mission["timestamp"])
                            if ((act_time - next_mission["timestamp"]) / 1000) > self.crossing_timeout:
                                print("Crossing timeout exceeded. Destroying mission.")
                                self.missions.pop(-1)
                    elif next_mission["mission"] == "interact":
                        self.missions.append(self.missions[0])
                        self.missions.pop(0)
                        self.abort(True)

        return self.monitor_affordance_state(target_affordance_node)

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
        todo_edge = self.graph.get_edge(self.robot_name, "aff_follow_" + str(self.person_id), "TODO")
        if todo_edge:
            try:
                self.graph.delete_edge(self.robot_name, "aff_follow_" + str(self.person_id), "TODO")
                self.graph.delete_edge("aff_follow_" + str(self.person_id),self.robot_name,  "TODO")
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