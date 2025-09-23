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

class InteractMission:
    def __init__(self, graph, agent_id, mission, start_timestamp):
        try:
            with open("src/main_prompts.json", "r", encoding="utf-8") as f:
                main_prompts = json.load(f)
                self.developer_prompt = main_prompts["interact"]
        except Exception as e:
            print(e)
            exit(0)

        print("Developer prompt", self.developer_prompt)

        self.start_timestamp = start_timestamp

        self.graph = graph
        self.mission = mission
        self.agent_id = agent_id
        self.data_history = deque(maxlen=3)  # Mantener un historial de mensajes
        self.ASR_words = deque(maxlen=3)
        self.data_for_dataset = []

        self.start_timestamp_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.LLM = LLMFunctions(self.developer_prompt)

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


        self.person_id = None

        self.mission_start_time = time.time()
        self.last_prompt_time_sent = 0
        self.prompt_sending_period = 1
        self.send_prompt_flag = False

        print("InteractMission initialized with graph and missions.")

    def __del__(self):
        print("InteractMission deleted.")

    def store_as_dataset(self):
        filename = f"dataset/prompt_{self.start_timestamp}.json"
        with open(filename, "w") as f:
            dict = {"messages" : list(self.data_for_dataset)}
            json.dump(dict, f, indent=4)

    def monitor(self):
        if time.time() - self.last_prompt_time_sent > self.prompt_sending_period:
            self.send_prompt_flag = True

        # if self.send_prompt_flag:
        # if self.send_prompt_flag or self.ASR_words:
        prompt_message, json_message = self.generate_state_data()
        if prompt_message is None or json_message is None:
            print("Data not valid")
            return {}
        self.data_for_dataset.append(json_message)
        self.data_history.append(prompt_message)
        response = {}
        temporal_series = "\n".join([m["content"] for m in list(self.data_history)])
        # print("Interact Temporal series:\n", temporal_series, "\n")
        # console.print(f"Interact temporal series: {temporal_series}", style='blue')
        fusioned_content = f"""{temporal_series}"""
        if self.last_response:
            fusioned_content += f"\n\n{self.last_response['content']}"

        user_prompt = {"role": "user", "content": fusioned_content}
        prompt_to_send = [user_prompt]
        valid, response = self.LLM.send_messages(prompt_to_send)
        if valid:
            response["timestamp"] = round(time.time() - self.start_timestamp, 2)
            # print("Respuesta interact mission:", response)
            console.print(f"Interact response: {response}", style='green')
            self.last_response = {"role": "user",
                                  "content": "Tu respuesta anterior ha sido '" + str(response) + "'"}
            # self.prompt_sending_period = float(response["time_next_inference"])

            self.last_prompt_time_sent = time.time()
            return response
        return {}

    def generate_state_data(self):
        # try:
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

            asr_words = f"Person told: {self.ASR_words.popleft()}" if len(self.ASR_words) > 0 else ""
            print("################## ASR WORDS:", asr_words)
            data_dict = {
                "person_name": person_node.name,
                "distance": person_pose,
                "orientation": angle_respect_to_robot,
                "time_last_event": time.time() - self.last_prompt_time_sent,
                "time_mission_start": (time.time() / 1000) - self.start_timestamp,
                "asr_words" : asr_words
            }
            state_description = self.LLM.describe_state(data_dict)
            sample_text = (
                "- Tiempo {time_mission_start}: "
                "{asr_words}"
                "{distance}, "
                "{orientation}. "
            ).format(
                time_mission_start=round(state_description["time_mission_start"], 2),
                distance=state_description["distance"],
                orientation=state_description["orientation"],
                asr_words=state_description["asr_words"],
            )
            return {"role": "user", "content": sample_text}, data_dict
        # except Exception as e:
        #     print(e)
        #     exit(0)

    def set_ASR_words(self, words):
        print("Setting ASR words:", words)
        self.ASR_words.append(words)
