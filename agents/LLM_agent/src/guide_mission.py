from collections import deque
import numpy as np
import time
import math
from .mission_base import MissionBase
from pydsr import Attribute


class GuideMission(MissionBase):
    def __init__(self, graph, LSTM_graph, agent_id, mission, start_timestamp, developer_prompt):
        super().__init__(graph, LSTM_graph, agent_id, mission, start_timestamp, developer_prompt)

        # Robot setup
        robot_nodes = self.graph.get_nodes_by_type("omnirobot") + self.graph.get_nodes_by_type("robot")
        if not robot_nodes: exit(0)
        robot_node = robot_nodes[0]
        self.robot_name = robot_node.name
        self.robot_id = robot_node.id

        self.allowed_elements = ["door", "person"]
        self.last_prompt_time_sent = 0
        self.prompt_sending_period = 2.5  # Give time to speak
        self.send_prompt_flag = True
        self.last_response = ""

        print("GuideMission (Simplified Text Mode) initialized.")

    def monitor(self):
        # Timer check
        # if time.time() - self.last_prompt_time_sent > self.prompt_sending_period:
        #     self.send_prompt_flag = True

        if not self.send_prompt_flag:
            return {}

        # 1. Generate the simple prompt text
        state_text = self.generate_state_data()
        print("Generated state data for prompt:")
        print(state_text)
        if not state_text:
            return {}

        # 2. Build messages
        user_prompt = {"role": "user", "content": state_text}
        prompt_to_send = [user_prompt]
        #
        # # Optional: Keep last robot saying for context
        # # if self.last_response:
        # #     prompt_to_send.append({"role": "assistant", "content": self.last_response})
        #
        if self.send_prompt_flag:
            # 3. Call LLM with require_json=False
            # print("Sending prompt to LLM...")
            valid, response_text = self.LLM.send_messages(prompt_to_send, require_json=False)

            if valid:
                # Clean up the text (remove quotes if LLM adds them)
                response_text = response_text.replace('"', '').strip()
                self.last_response = response_text

                print(f"ROBOT SAYS: {response_text}")

                # 4. Send to TTS
                tts_node = self.graph.get_node("TTS")
                if tts_node:
                    tts_node.attrs["words_to_say"] = Attribute(str(response_text), self.agent_id)
                    self.graph.update_node(tts_node)

                # Calculate wait time (approx 0.4s per word)
                wait_time = len(response_text.split()) * 0.4 + 1.5
                self.prompt_sending_period = wait_time

            else:
                print("LLM failed to respond.")
                self.prompt_sending_period = 3.0

            self.last_prompt_time_sent = time.time()
            self.send_prompt_flag = False

            return {"response": self.last_response}

        return {}

    def generate_state_data(self):
        try:
            # --- Get Person ---
            person_node = self.graph.get_node(self.mission["person"])
            if not person_node: return None

            # --- Get Target ---
            target_name = self.mission.get("target", "unknown_target")
            target_node = self.graph.get_node(target_name)
            if not target_node: return None
            target_description = f"{target_node.type}"
            if target_node.type == "door":
                if "connected_room_name" in target_node.attrs:
                    connected_room_name = target_node.attrs["connected_room_name"].value
                    target_description += f" to {connected_room_name}"

            # -- Get target location respect to person ---
            target_pose_vector = self.inner_api.get_translation_vector(person_node.name, target_name, np.iinfo(np.int64).max)
            target_pose = self._get_semantic_location(target_pose_vector[0], target_pose_vector[1])

            # --- Get Person Location (Semantic) ---
            person_robot_pose_matrix = self.inner_api.get_transformation_matrix(self.robot_name, person_node.name)
            px = person_robot_pose_matrix[0, 3] / 1000
            py = person_robot_pose_matrix[1, 3] / 1000
            # person_loc = self._get_semantic_location(px, py)  # Helper function

            # --- Sort Elements (FOV vs Non-FOV) ---
            visible_items = []
            hidden_items = []

            robot_node = self.graph.get_node(self.robot_id)

            for node in self.graph.get_nodes():
                if node.type in self.allowed_elements and node.id != person_node.id and node.id != robot_node.id and node.id != target_node.id:
                    print(f"Processing node: {node.name} of type {node.type}")
                    # Relative position to PERSON (important for guidance)
                    vec = self.inner_api.get_translation_vector(person_node.name, node.name, np.iinfo(np.int64).max)
                    sem_pos = self._get_semantic_location(vec[0], vec[1])
                    element_description = f"{node.type}"
                    if node.type == "door":
                        if "connected_room_name" in node.attrs:
                            connected_room_name = node.attrs["connected_room_name"].value
                            element_description += f" to {connected_room_name}"


                    # Check FOV edge
                    if self.graph.get_edge(person_node.id, node.id, "in_FOV"):
                        visible_items.append({
                        "name": element_description,
                        "pose_respect_to_person": sem_pos,
                        # "proximity_to_goal_door": proximity
                    })
                    else:
                        hidden_items.append({
                        "name": element_description,
                        "pose_respect_to_person": sem_pos,
                        # "proximity_to_goal_door": proximity
                    })

            # --- Construct Prompt ---
            prompt = str({
        "target": target_description,
        "target_relative_pose": target_pose,
        # "person_behavior": person_behavior,
        "visible_landmarks": visible_items
            })
            return prompt

        except Exception as e:
            print(f"Error generating state: {e}")
            return None

    def _get_semantic_location(self, x, y):
        """
        Calculates a natural language description combining distance and angle.
        Coordinate assumption: +Y is Front, +X is Right.
        """
        x /= 1000  # Convert mm to meters
        y /= 1000  # Convert mm to meters

        # 1. Distance Logic
        distance = math.sqrt(x ** 2 + y ** 2)

        if distance < 1.5:
            dist_str = "very close"
        elif distance < 3.5:
            dist_str = "close"
        elif distance < 6.0:
            dist_str = "nearby"
        else:
            dist_str = "far"

        # 2. Angle Logic (Sectors)
        # atan2(x, y) gives angle relative to Forward (Y-axis)
        # 0 = Front, 90 = Right, -90 = Left, 180/-180 = Back
        angle_deg = math.degrees(math.atan2(x, y))

        if -22.5 <= angle_deg <= 22.5:
            dir_str = "directly in front"
        elif 22.5 < angle_deg <= 67.5:
            dir_str = "in front to the right"
        elif 67.5 < angle_deg <= 112.5:
            dir_str = "to the right"
        elif 112.5 < angle_deg <= 157.5:
            dir_str = "behind to the right"
        elif angle_deg > 157.5 or angle_deg < -157.5:
            dir_str = "directly behind"
        elif -157.5 <= angle_deg < -112.5:
            dir_str = "behind to the left"
        elif -112.5 <= angle_deg < -67.5:
            dir_str = "to the left"
        elif -67.5 <= angle_deg < -22.5:
            dir_str = "in front to the left"
        else:
            dir_str = "nearby"

        return f"{dist_str} {dir_str}"

    def handle_graph_event(self, event_batch):
        # Just trigger an immediate update if something happens
        self.send_prompt_flag = True
        self.last_prompt_time_sent = 0

    def enable_sending_prompts(self):
        self.send_prompt_flag = True
        self.last_prompt_time_sent = 0