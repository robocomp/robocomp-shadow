from pydsr import *
from src.LLM_aux_functions import LLMFunctions
from collections import deque
import numpy as np
import time
import json
import os
from datetime import datetime

from rich.console import Console

console = Console(highlight=False)

class FollowMission:
    def __init__(self, graph, agent_id, mission, start_timestamp):

        try:
            with open("src/main_prompts.json", "r", encoding="utf-8") as f:
                main_prompts = json.load(f)
                self.developer_prompt = main_prompts["follow"]
        except Exception as e:
            print(e)
            exit(0)

        print("Developer prompt", self.developer_prompt)


        self.start_timestamp = start_timestamp

        self.graph = graph
        self.mission = mission
        self.agent_id = agent_id
        self.messages_history = deque(maxlen=10)  # Mantener un historial de mensajes
        self.data_history = deque(maxlen=5)  # Mantener un historial de mensajes
        self.data_for_dataset = []

        self.start_timestamp_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.LLM = LLMFunctions(self.developer_prompt)

        self.current_bt_state = None

        self.distance = deque(maxlen=12)
        self.pos = deque(maxlen=10)
        self.lambda_ = deque(maxlen=10)
        self.speed = deque(maxlen=10)
        self.intentions = deque(maxlen=10)

        self.rt_api = rt_api(self.graph)
        self.inner_api = inner_api(self.graph)

        self.last_response = None

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
        self.initialized = False

        self.mission_start_time = time.time()
        self.last_prompt_time_sent = 0
        self.prompt_sending_period = 1
        self.send_prompt_flag = False

        print("FollowMission initialized with graph and missions.")

    def __del__(self):
        print("FollowMission deleted.")

    def store_as_dataset(self):
        filename = f"dataset/prompt_{self.start_timestamp}.json"
        with open(filename, "w") as f:
            dict = {"messages" : list(self.data_for_dataset)}
            json.dump(dict, f, indent=4)

    def monitor(self):
        if time.time() - self.last_prompt_time_sent > self.prompt_sending_period:
            self.send_prompt_flag = True

        if self.send_prompt_flag:
            prompt_message, json_message = self.generate_state_data()
            if prompt_message is None or json_message is None:
                print("Data not valid")
                return
            self.data_for_dataset.append(json_message)
            self.data_history.append(prompt_message)
            # print("############ DATA HISTORY", self.data_history)
            if not self.initialized:
                user_prompt = {"role": "user",
                     "content": "Tell the person you are going to start following."}
                self.initialized = True
            else:
                temporal_series = "\n".join([m["content"] for m in list(self.data_history)])
                console.print(f"Interact temporal series: {temporal_series}", style='blue')
                user_prompt = {"role": "user", "content": f"""
{temporal_series}
                                """}
            prompt_to_send = [user_prompt]
            if self.last_response:
                prompt_to_send.append(self.last_response)
            # print("prompt_to_send", prompt_to_send)
            valid, response = self.LLM.send_messages(prompt_to_send)
            if valid:
                response["timestamp"] = round(time.time() - self.start_timestamp, 2)
                print("Respuesta LLM:", response)

                self.prompt_sending_period = float(response["time_next_inference"])
                self.last_prompt_time_sent = time.time()
                return response

            self.last_prompt_time_sent = time.time()
            self.send_prompt_flag = False

    def generate_state_data(self):
        try:
            person_node = self.graph.get_node(self.mission["target"])
            if person_node is None:
                print(f"Person node {self.mission['target']} not found in the graph.")
                return None, None
            self.person_id = person_node.attrs["person_id"].value
            if self.person_id:
                # Check if person if lost
                is_person_lost = person_node.attrs["out_of_robot_FOV"].value
                if not is_person_lost:
                    # Get person pose respect to robot
                    person_robot_pose_matrix = self.inner_api.get_transformation_matrix(self.robot_name, person_node.name)
                    person_pose = (person_robot_pose_matrix[0, 3] / 1000, person_robot_pose_matrix[1, 3] / 1000)
                    angle_respect_to_robot = round(
                        np.arctan2(person_robot_pose_matrix[1, 0], person_robot_pose_matrix[0, 0]), 2)
                else:
                    person_pose = None
                    angle_respect_to_robot = None

                data_dict = {
                    "person_name" : person_node.name,
                    "distance" : person_pose,
                    "orientation" : angle_respect_to_robot,
                    "time_last_event" : time.time() - self.last_prompt_time_sent,
                    "time_mission_start" : time.time() - self.mission_start_time
                }

                state_description = self.LLM.describe_state(data_dict)

                sample_text = (
                    "- Tiempo {time_mission_start}: "
                    "{distance}, "
                    "{orientation}. "
                ).format(
                    time_mission_start=round(state_description["time_mission_start"], 2),
                    distance=state_description["distance"],
                    orientation=state_description["orientation"],
                )
                return {"role": "user", "content": sample_text}, data_dict

        except Exception as e:
            print("Error reading person pose. Possible graph changes")
            return None, None