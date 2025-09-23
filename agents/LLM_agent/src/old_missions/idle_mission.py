from pydsr import *
from src.LLM_aux_functions import LLMFunctions
from collections import deque
import numpy as np
import time
import json
import ast
import os
from datetime import datetime
from rich.console import Console

console = Console(highlight=False)

class IdleMission:
    def __init__(self, graph, agent_id, missions):
        try:
            with open("src/main_prompts.json", "r", encoding="utf-8") as f:
                main_prompts = json.load(f)
                self.developer_prompt = main_prompts["idle"]
        except Exception as e:
            print(e)
            exit(0)

        print("Developer prompt", self.developer_prompt)

        self.missions = missions
        self.graph = graph
        self.agent_id = agent_id
        self.messages_history = deque(maxlen=10)  # Mantener un historial de mensajes
        self.data_history = deque(maxlen=1)  # Mantener un historial de mensajes
        self.data_for_dataset = []

        self.start_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
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

        current_edges = self.graph.get_edges_by_type("current")
        if current_edges:
            actual_room = self.graph.get_node(current_edges[0].destination)
            actual_room_id = actual_room.attrs["room_id"].value
            if actual_room_id is not None:
                self.actual_room_id = actual_room_id

        self.person_id = None

        self.mission_start_time = time.time()
        self.last_prompt_time_sent = 0
        self.prompt_sending_period = 1
        self.send_prompt_flag = False

        print("IdleMission initialized with graph and missions.")

    def __del__(self):
        print("IdleMission deleted.")

    def store_as_dataset(self):
        filename = f"dataset/prompt_{self.start_timestamp}.json"
        with open(filename, "w") as f:
            dict = {"messages" : list(self.data_for_dataset)}
            json.dump(dict, f, indent=4)

    def monitor(self, active_mission_result):
        # if time.time() - self.last_prompt_time_sent > self.prompt_sending_period:
        #     self.send_prompt_flag = True

        prompt_message, json_message = self.generate_state_data()
        if prompt_message is None or json_message is None:
            print("Data not valid")
            return
        self.data_for_dataset.append(json_message)
        self.data_history.append(prompt_message)
        # if self.send_prompt_flag:
        temporal_series = "\n".join([m["content"] for m in list(self.data_history)])
        # print("Idle Temporal series:\n", temporal_series, "\n")
        fusioned_content = f"""{temporal_series}"""

        if self.last_response:
            fusioned_content += f"\n- La respuesta anterior del supervisor ha sido: {self.last_response}"

        if active_mission_result:
            fusioned_content += f"\n- La respuesta anterior del LLM de la misión activa ha sido: {active_mission_result}"
        console.print(f"Supervisor data: {fusioned_content}", style='blue')
        user_prompt = {"role": "user", "content": fusioned_content}
        prompt_to_send = [user_prompt]
        # print("prompt_to_send", prompt_to_send)
        valid, response = self.LLM.send_messages(prompt_to_send)
        # console.print(f"Supervisor original response: {response}", style='yellow')
        if valid:
            llm_node = self.graph.get_node("LLM")
            if llm_node is not None:
                llm_node.attrs["LLM_response"] = Attribute(str(response), self.agent_id)
                self.graph.update_node(llm_node)
            # print("Respuesta Idle:", response)
            response["timestamp"] = round(time.time() - self.mission_start_time, 2)
            console.print(f"Supervisor response: {response}", style='red')
            self.last_response = str(response)
            self.prompt_sending_period = float(response["time_next_inference"])
        # else:
            # self.messages_history.append(
            #     {"role": "assistant", "content": response["to_say"]})
        self.last_prompt_time_sent = time.time()
        self.send_prompt_flag = False

    def generate_state_data(self):
        # try:
            # Check robot battery data
        robot_node = self.graph.get_node(self.robot_id)
        if robot_node is None:
            print("Robot node not found in the graph.")
            return None, None
        robot_speed_adv = robot_node.attrs["robot_current_advance_speed"].value
        robot_speed_rot = robot_node.attrs["robot_current_angular_speed"].value

        battery_node = self.graph.get_node("battery")
        if battery_node is None:
            print("Battery node not found in the graph.")
            return None, None
        battery_load = battery_node.attrs["battery_load"].value

        # Check current robot room
        room_name = ""
        current_edges = self.graph.get_edges_by_type("current")
        if not current_edges:
            print("No current edges found in the graph.")
        room_name = self.graph.get_node(current_edges[0].destination).name

        robot_mission_nodes = self.graph.get_nodes_by_type("mission")
        actual_missions = [ast.literal_eval(robot_mission_node.attrs["plan"].value)[0] for robot_mission_node in robot_mission_nodes]

        # print("Acutal missions:", actual_missions)

        # Check if robot is actually executing any mission
        robot_actual_submissions = [self.graph.get_node(edge.destination) for edge in
                                   self.graph.get_edges_by_type("TODO") if edge.origin == self.robot_id]

        robot_submissions = [(submission.name, self.graph.get_node(submission.attrs["parent"].value).name) for
                             submission in robot_actual_submissions]

        people_with_intentions_to_robot = [self.graph.get_node(edge.origin).name for edge in self.graph.get_edges_by_type("has_intention") if edge.destination == self.robot_id]

        data_dict = {
            "robot_speed" : [robot_speed_adv, robot_speed_rot],
            "people_with_intentions_to_robot" : people_with_intentions_to_robot,
            "robot_submissions" : robot_submissions,
            "actual_room_name" : room_name,
            "battery_load" : battery_load,
            "robot_actual_missions" : actual_missions,
            "time_mission_start": time.time() - self.mission_start_time
        }

        state_description = self.LLM.describe_state(data_dict)

        sample_text = (
            "- Tiempo {time_mission_start}: "
            "Shadow está {robot_speed}. "
            "Nivel de batería al {battery_load}%. "
            "Shadow está en {actual_room_name}. "
            "{robot_actual_missions}. "
            "{robot_submissions}. "
            "{people_with_intentions_to_robot}"
        ).format(
            time_mission_start=round(state_description["time_mission_start"], 2),
            robot_speed=state_description["robot_speed"],
            battery_load=state_description["battery_load"],
            actual_room_name=state_description["actual_room_name"],
            robot_submissions=(
                "El affordance que Shadow está ejecutando actualmente en la misión es "
                + " ".join(state_description["robot_submissions"][0])
                if state_description["robot_submissions"]
                else "No estás ejecutando ningún affordance"
            ),
            robot_actual_missions=(
                "Shadow está ejecutando la misión " + state_description["robot_actual_missions"][0]["mission"] + " " + state_description["robot_actual_missions"][0]["target"]
                if state_description["robot_actual_missions"]
                else "Shadow no está ejecutando ninguna misión"
            ),
            people_with_intentions_to_robot=f"{state_description['people_with_intentions_to_robot'][0]} parece que quiere interactuar contigo." if state_description["people_with_intentions_to_robot"] else "No hay personas con intención de interactuar contigo."
        )
        return {"role": "user", "content": sample_text}, data_dict

        # except Exception as e:
        #     print("Error reading person pose. Possible graph changes")
        #     return None, None
