from pydsr import *
from collections import deque
from .mission_base import MissionBase  # Assuming MissionBase is in a separate file


class FollowMission(MissionBase):
    def __init__(self, graph, mission, agent_id):
        super().__init__(graph, mission, agent_id)

        # Class Constants
        self.CROSSING_TIMEOUT = 5
        self.PERSON_OUT_MAX_TIMES = 10

        self.robot_resources = ["base", "speaker"]
        self.inner_api = inner_api(self.graph)
        self.current_affordance_target_name = None
        self.person_out_times = 0

        # Initial checks
        if not self.missions:
            print("No missions provided. Exiting.")
            return

        person_node = self.graph.get_node(self.missions[0]["target"])
        if not person_node:
            print(f"Person node {self.missions[0]['target']} not found in the graph.")
            return

        self.person_id = person_node.attrs["person_id"].value
        target_affordance_node = self.get_current_target_affordance_node(self.missions[0]["target"],
                                                                         self.missions[0]["mission"])
        self.insert_TODO_edge(target_affordance_node)
        print("GOTOMission initialized with graph and missions.")

    def monitor(self):
        if not self.missions:
            print("No missions to monitor. Mission finished.")
            self.current_bt_state = None
            return True

        self.current_mission = self.missions[0]
        target_affordance_node = self.get_current_target_affordance_node(self.current_mission["target"],
                                                                         self.current_mission["mission"])
        if not target_affordance_node:
            return False

        if self._check_and_complete_mission(target_affordance_node):
            return True

        print("Current missions:", self.missions)
        #
        # if self.current_mission['mission'] == "follow":
        #     if len(self.missions) > 1:
        #         next_mission = self.missions[1]
        #         if next_mission["mission"] == "cross":
        #             self._check_crossing_behavior(next_mission)
        #         elif next_mission["mission"] == "interact":
        #             print("Interact mission appended. Stopping following for now.")
        #             self.abort_and_store_current_submission()

        return False

    def _check_and_complete_mission(self, affordance_node):
        is_active = affordance_node.attrs["active"].value
        bt_state = affordance_node.attrs["bt_state"].value

        if is_active is False and bt_state == "completed":
            print(f"Mission {affordance_node.name} completed. Removing TODO edge.")
            self.remove_TODO_edge(affordance_node)
            self.missions.popleft()
            self.current_bt_state = None
            return True
        return False

    def _check_crossing_behavior(self, next_mission):
        robot_node = self.graph.get_node(self.robot_id)
        if not robot_node: return

        act_time = robot_node.attrs["timestamp_alivetime"].value
        person_pose_respect_to_door = self.inner_api.get_translation_vector(next_mission['target'],
                                                                            self.current_mission['target'])

        if person_pose_respect_to_door and person_pose_respect_to_door[1] > 100:
            self.person_out_times += 1
            if self.person_out_times >= self.PERSON_OUT_MAX_TIMES:
                print("Person is out of the door for too long. Changing mission to cross affordance.")
                self.abort_and_store_current_submission()
                self.person_out_times = 0
        else:
            self.person_out_times = 0
            if (act_time - next_mission["timestamp"]) / 1000 > self.CROSSING_TIMEOUT:
                print("Crossing timeout exceeded. Destroying cross mission.")
                self.missions.remove(next_mission)

