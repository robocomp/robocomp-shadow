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
from src.long_term_graph import LongTermGraph

console = Console(highlight=False)

class IdleMission:
    def __init__(self, graph, agent_id, missions, start_timestamp):
        try:
            with open("src/main_prompts.json", "r", encoding="utf-8") as f:
                main_prompts = json.load(f)
                self.developer_prompt = main_prompts["idle"]
        except Exception as e:
            print(e)
            exit(0)

        print("Developer prompt", self.developer_prompt)

        self.start_timestamp = start_timestamp
        self.missions = missions
        self.graph = graph
        self.agent_id = agent_id
        self.messages_history = deque(maxlen=10)  # Mantener un historial de mensajes
        self.data_history = deque(maxlen=1)  # Mantener un historial de mensajes
        self.data_for_dataset = []

        home_dir = os.path.expanduser("~")
        self.graph_path = os.path.join(home_dir, "igraph_LTSM", "graph.pkl")
        self.long_term_graph = LongTermGraph(self.graph_path)

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

        current_edges = self.graph.get_edges_by_type("current")
        if current_edges:
            actual_room = self.graph.get_node(current_edges[0].destination)
            actual_room_id = actual_room.attrs["room_id"].value
            if actual_room_id is not None:
                self.actual_room_id = actual_room_id

        self.person_id = None

        self.mission_start_time = time.time()
        self.last_prompt_time_sent = 0
        self.send_prompt_flag = False

        self.element_node_attributes_to_avoid = ["ID", "level", "parent", "pos_x", "pos_y", "room_id", "lambda_cont"]
        self.element_node_types_to_avoid = ["room", "robot", "omnirobot", "wall"]

        print("IdleMission initialized with graph and missions.")

    def __del__(self):
        print("IdleMission deleted.")

    def store_as_dataset(self):
        filename = f"dataset/prompt_{self.start_timestamp}.json"
        with open(filename, "w") as f:
            dict = {"messages" : list(self.data_for_dataset)}
            json.dump(dict, f, indent=4)

    def monitor(self, active_mission_result):
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
            for val in dict(self.last_response):  # self.last_response is dict_items
                fusioned_content += f"\n    - {val} ---> {self.last_response[val]}"

        if active_mission_result:
            fusioned_content += f"\n[MISSION-SPECIFIC LLM LAST RESPONSE]"
            for val in dict(active_mission_result):
            #     fusioned_content += f"\n- Last supervisor LLM (yours) response {key} was: {value}"
                fusioned_content += f"\n    - {val} ---> {active_mission_result[val]}"
        console.print(f"Supervisor data: {fusioned_content}", style='yellow')
        user_prompt = {"role": "user", "content": fusioned_content}
        prompt_to_send = [user_prompt]
        # print("prompt_to_send", prompt_to_send)
        start_time = time.time()
        valid, response = self.LLM.send_messages(prompt_to_send)
        end_time = time.time()
        print(f"LLM response time: {end_time - start_time} seconds")
        # console.print(f"Supervisor original response: {response}", style='yellow')
        if valid:
            llm_node = self.graph.get_node("LLM")
            if llm_node is not None:
                llm_node.attrs["LLM_response"] = Attribute(str(response), self.agent_id)
                self.graph.update_node(llm_node)
            # print("Respuesta Idle:", response)
            response["timestamp"] = round(time.time() - self.start_timestamp, 2)
            console.print(f"Supervisor response: {response}", style='red')
            self.last_response = response
        # else:
            # self.messages_history.append(
            #     {"role": "assistant", "content": response["to_say"]})
        self.last_prompt_time_sent = time.time()
        self.send_prompt_flag = False

    def generate_state_data(self):
        robot_data = self.get_robot_data()
        rooms_data = self.get_rooms_data()
        missions_data = self.get_missions_data()
        elements_data = self.get_elements_data(rooms_data)
        # print("Robot data:", robot_data)
        # print("Rooms data:", rooms_data)
        # print("Missions data:", missions_data)
        # print("Elements data:", elements_data)

        scene_description = self.tabulate_scene_with_relations(robot_data, rooms_data, missions_data, elements_data)

        # people_with_intentions_to_robot = [self.graph.get_node(edge.origin).name for edge in self.graph.get_edges_by_type("has_intention") if edge.destination == self.robot_id]

        # data_dict = {
        #     "robot_speed" : [robot_speed_adv, robot_speed_rot],
        #     "people_with_intentions_to_robot" : people_with_intentions_to_robot,
        #     "robot_submissions" : robot_submissions,
        #     "actual_room_name" : room_name,
        #     "battery_load" : battery_load,
        #     "robot_actual_missions" : actual_missions,
        #     "time_mission_start": time.time() - self.mission_start_time
        # }
        #
        # state_description = self.LLM.describe_state(data_dict)
        data_dict = {}
        # sample_text = (
        #     "- Tiempo {time_mission_start}: "
        #     "Shadow está {robot_speed}. "
        #     "Nivel de batería al {battery_load}%. "
        #     "Shadow está en {actual_room_name}. "
        #     "{robot_actual_missions}. "
        #     "{robot_submissions}. "
        #     "{people_with_intentions_to_robot}"
        # )
        # sample_text = (
        #     "- Tiempo {time_mission_start}: "
        #     "Shadow está {robot_speed}. "
        #     "Nivel de batería al {battery_load}%. "
        #     "Shadow está en {actual_room_name}. "
        #     "{robot_actual_missions}. "
        #     "{robot_submissions}. "
        #     "{people_with_intentions_to_robot}"
        # ).format(
        #     time_mission_start=round(state_description["time_mission_start"], 2),
        #     robot_speed=state_description["robot_speed"],
        #     battery_load=state_description["battery_load"],
        #     actual_room_name=state_description["actual_room_name"],
        #     robot_submissions=(
        #         "El affordance que Shadow está ejecutando actualmente en la misión es "
        #         + " ".join(state_description["robot_submissions"][0])
        #         if state_description["robot_submissions"]
        #         else "No estás ejecutando ningún affordance"
        #     ),
        #     robot_actual_missions=(
        #         "Shadow está ejecutando la misión " + state_description["robot_actual_missions"][0]["mission"] + " " + state_description["robot_actual_missions"][0]["target"]
        #         if state_description["robot_actual_missions"]
        #         else "Shadow no está ejecutando ninguna misión"
        #     ),
        #     people_with_intentions_to_robot=f"{state_description['people_with_intentions_to_robot'][0]} parece que quiere interactuar contigo." if state_description["people_with_intentions_to_robot"] else "No hay personas con intención de interactuar contigo."
        # )
        return {"role": "user", "content": scene_description}, data_dict

        # except Exception as e:
        #     print("Error reading person pose. Possible graph changes")
        #     return None, None

    def get_robot_data(self):
        robot_node = self.graph.get_node(self.robot_id)
        robot_speed_adv = 0
        robot_speed_rot = 0
        battery_load = -99
        if robot_node is not None:
            robot_speed_adv = robot_node.attrs["robot_current_advance_speed"].value
            robot_speed_rot = robot_node.attrs["robot_current_angular_speed"].value
            battery_node = self.graph.get_node("battery")
            if battery_node is None:
                print("Battery node not found in the graph.")
                return None, None
            battery_load = battery_node.attrs["battery_load"].value
            battery_is_charging = battery_node.attrs["is_charging"].value
            return {"advance_speed": round(robot_speed_adv, 2), "rotational_speed": round(robot_speed_rot, 2), "battery_load": round(battery_load, 2), "is_charging": battery_is_charging}

    def get_rooms_data(self):
        # Check current robot room
        actual_room_name = ""
        current_edges = self.graph.get_edges_by_type("current")
        if current_edges:
            actual_room_name = self.graph.get_node(current_edges[0].destination).name
        ltsm_rooms = self.long_term_graph.get_igraph_rooms()
        return {"current_room": actual_room_name, "known_rooms": ltsm_rooms}

    def get_missions_data(self):
        available_missions = ["goto", "follow", "interact"]
        robot_mission_nodes = [node for node in self.graph.get_nodes_by_type("mission") if node.name != "mission_history"]
        robot_current_submissions_parsed, robot_mission_nodes_dict = [], []
        if robot_mission_nodes:
            robot_mission_nodes_dict = [{"mission" : robot_mission_node.attrs["plan_type"].value, "submissions" : [submission for submission in ast.literal_eval(robot_mission_node.attrs["plan"].value)], "target" : robot_mission_node.attrs["plan_target"].value} for robot_mission_node in robot_mission_nodes]

            # Check if robot is actually executing any mission
            robot_current_submissions = [self.graph.get_node(edge.destination) for edge in
                                       self.graph.get_edges_by_type("TODO") if edge.origin == self.robot_id]

            robot_current_submissions_parsed = [(submission.name, self.graph.get_node(submission.attrs["parent"].value).name) for
                                 submission in robot_current_submissions]
        return {"available_missions": available_missions, "robot_current_missions": robot_mission_nodes_dict, "robot_current_submissions": robot_current_submissions_parsed}

    def get_elements_data(self, rooms_data):
        elements_data = {}
        current_edges = self.graph.get_edges_by_type("current")
        if current_edges:
            elements_data[rooms_data["current_room"]] = []
            actual_room = self.graph.get_node(rooms_data["current_room"])
            actual_room_edges = actual_room.get_edges()
            elements_in_room = [self.graph.get_node(edge.destination) for edge in actual_room_edges]
            for element in elements_in_room:
                if element is not None:
                    if not element.type in self.element_node_types_to_avoid:
                        new_element = {"data" : {"name" : element.name, "type" : element.type}, "children" : []}
                        # Insert in new element attributes from element.attrs avoiding the ones in self.element_node_attributes_to_avoid
                        for attr in element.attrs:
                            if not attr in self.element_node_attributes_to_avoid:
                                new_element["data"][attr] = element.attrs[attr].value
                        element_child_nodes = [(self.graph.get_node(edge.destination), edge.type) for edge in element.get_edges()]
                        element_child_data = [{"name": node[0].name, "type": node[0].type, "relationship": node[1]} for node in element_child_nodes if node[0] is not None]
                        new_element["children"] = element_child_data
                        elements_data[rooms_data["current_room"]] .append(new_element)
        for known_room in rooms_data["known_rooms"]:
            if known_room != rooms_data["current_room"]:
                print("Checking known room:", known_room)
                elements_data[known_room] = []
                elements_in_room = self.long_term_graph.get_igraph_elements_given_room_name(known_room)
                for element in elements_in_room:
                    print("Element in LTSM:", element)

        return elements_data


    def tabulate_scene_with_relations(self, robot_data, rooms_data, missions_data, elements_data):
        lines = []

        current_room = rooms_data.get("current_room")
        known_rooms = rooms_data.get("known_rooms", [])

        lines.append(f"[TIMESTAMP] {time.time() - self.start_timestamp}")
        for room_name in known_rooms:
            room_marker = "[ROOM]"
            if room_name == current_room:
                room_marker += " (current)"
            lines.append(f"{room_marker} {room_name}")

            # Robot in current room
            if room_name == current_room:
                battery = robot_data.get("battery_load", 0.0)
                lines.append(
                    f"    - contains ROBOT (battery={battery} {'charging' if robot_data['is_charging'] else 'not charging'}, advance_speed={robot_data.get('advance_speed', 0)}, rotational_speed={robot_data.get('rotational_speed', 0)})")

            # Elements in current room
            for elem in elements_data[room_name]:
                elem_data = elem.get("data", {})
                elem_type = elem_data.get("type", "unknown")
                elem_name = elem_data.get("name", "unknown")

                # Only include elements in current room
                if not elem_type in self.element_node_types_to_avoid:
                    attrs = ", ".join([f"{k}={v}" for k, v in elem_data.items() if k not in self.element_node_attributes_to_avoid])
                    attr_str = f" [{attrs}]" if attrs else ""
                    lines.append(f"    - contains ELEMENT {elem_name} [{elem_type}]{attr_str}")

                    # Affordances
                    for child in elem.get("children", []):
                        if child.get("type") == "affordance":
                            lines.append(f"        - affordance: {child.get('name')}")

        # Missions
        available_missions = missions_data.get("available_missions", [])
        lines.append("\n[AVAILABLE MISSIONS]")
        if available_missions:
            for m in available_missions:
                lines.append(f"    - {m}")
        else:
            lines.append("    - None (robot has no available missions)")

        robot_current_missions = missions_data.get("robot_current_missions", [])
        lines.append("\n[ACTIVE MISSIONS]")
        if robot_current_missions:
            for m in robot_current_missions:
                current_submission = {}
                if m["submissions"]:
                    current_submission = m["submissions"][0]
                lines.append(f"    - {m['mission']} ---> {m['target']}. Current submission in mission: {current_submission['mission']} ---> {current_submission['target']}" if current_submission != {} else "None")
        else:
            lines.append("    - None (robot has no active missions)")

        # Relations
        lines.append("\n[RELATIONS]")
        for elem in elements_data[current_room]:
            elem_name = elem.get("data", {}).get("name")
            for child in elem.get("children", []):
                if child.get("relationship") in ["has_intention", "TODO"]:
                    lines.append(f"    - {elem_name} --{child.get('relationship')}--> {child.get('name')}")

        lines.append("\n[YOUR LAST RESPONSE]")
        return "\n".join(lines)
