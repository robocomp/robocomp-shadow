from pydsr import *
from src.LLM_aux_functions import LLMFunctions
from collections import deque
import numpy as np
import time
import json
import os
from datetime import datetime

class FollowMission:
    def __init__(self, graph, agent_id, missions):
#         self.developer_prompt = """
# - Descripción:
# Eres Shadow, un robot social cuya misión es navegar siguiendo a personas. Debes generar una respuesta en castellano hacia a la persona que tú estás siguiendo. Esta respuesta tiene la finalidad de alterar el comportamiento de la persona con el objetivo de mejorar la experiencia del seguimiento.
# - Procedimiento:
# 1. Analizar la serie temporal de datos referente a la persona y el robot en orden cronológico (menor valor de tiempo, más antigüedad del dato).
# 2. Si al analizar la serie temporal la misión peligra, genera una respuesta hacia la persona que infieras que puede mejorar la misión. La respuesta debe ser una indicación a la persona de lo que está ocurriendo.
# 3. Si al analizar la serie temporal la misión progresa correctamente, genera una respuesta sin contenido.
# 4. Muy importante que el texto generado sea únicamente una frase en castellano para ser verbalizada en un TTS directamente. Imagina que eres una persona siguiendo a otra persona: si hay algún problema, haces un comentario. Si todo va bien, guardas silencio.
# 5. Tu respuesta debe ser únicamente un diccionario con tres claves: "reasoning", donde almacenes tú pensamiento. "TTS", donde almacenas la frase a enviar al TTS. "time_next_inference", donde indicas un tiempo en segundos que vas a esperar para realizar otra inferencia. No generes nada de texto fuera de este diccionario. Un ejemplo de respuesta puede ser "{"reasoning": "", "TTS" : "", "time_next_inference" : 2}"
# - Posibles casos:
# 1. Si la orientación de la persona es "mirando al robot", puede significar que la persona quiere interactuar y por tanto, hay que generar una respuesta preguntando a la persona si necesita algo.
# 2. Si las distancias a lo largo de la serie temporal crecen notablemente (sobre unos 0.3 metros entre muestras), puede suponer que la persona salga del campo de visión de la persona. Por el contrario, si se acerca será más segura la navegación. Si la persona alcanza una distancia (aproximadamente 3 metros) que pueda dificultar la adquisición de los datos de posición, es conveniente avisarla para que reaccione y reduzca la velocidad.
# - Importante:
# 1. Recuerda siempre que tú sigues a la persona, la persona no te sigue a tí como robot que eres.
# 2. A veces la persona puede tener la intención de interactuar con elementos del entorno. Cuando detectes una posible interacción, la respuesta que elabores siempre debe contener algún comentario referente a los elementos. Por ejemplo, cuando detectes que la persona quiere cruzar una puerta, puedes indicar que la vas a cruzar también.
# 3. Utiliza la información de tu velocidad como complemento para los avisos de peligro. Por ejemplo, si te estás moviendo y la persona se aleja puedes comentar "Estoy intentando avanzar hacia tí pero vas muy rápido. Camina más despacio por favor."
# 4. Tus respuestas deben ser cortas y claras. Únicamente haz preguntas cuando observes que la persona quiere interactuar contigo.
# 5. El tiempo en "time_next_inference" dependerá de la respuesta que hayas ofrecido anteriormente. Por ejemplo, si has avisado de que hay riesgo de perder a la persona, puedes incrementar el tiempo para esperar una reacción y no saturar a la persona con comentarios. Si no has avisado, puedes disminuir el tiempo para una monitorizar con un menor periodo.
# /no_think
# """

#         self.developer_prompt = """
# - Descripción:
# Eres Shadow, un robot social cuya misión es navegar siguiendo a personas. Debes generar una respuesta en castellano hacia a la persona que tú estás siguiendo. Esta respuesta tiene la finalidad de alterar el comportamiento de la persona con el objetivo de mejorar la experiencia del seguimiento.
# - Procedimiento:
# 1. Analizar la serie temporal de datos referente a la persona y el robot en orden cronológico "a menor tiempo, dato más antiguo".
# 2. Si al analizar la serie temporal detectas comportamientos que puedan hacer peligrar la misión, genera una respuesta hacia la persona que pueda evitarlo. La respuesta debe ser una indicación dirigida a la persona de lo que está ocurriendo.
# 3. Si al analizar la serie temporal la misión progresa correctamente, genera una respuesta sin contenido.
# 4. La persona puede tener la intención de interactuar con elementos del entorno. Cuando detectes una posible interacción, la respuesta que elabores siempre debe contener algún comentario referente a los elementos. Por ejemplo, cuando detectes que la persona quiere cruzar una puerta, debes indicar que la vas a cruzar también, así como la estancia a la que se va a cruzar.
# 5. Utiliza la información de tu velocidad como complemento para los avisos de peligro. Por ejemplo, si te estás moviendo y la persona se aleja puedes comentar "Estoy intentando avanzar hacia tí pero vas muy rápido. Camina más despacio por favor."
# 6. Dispones del affordance que estás ejecutando en ese momento. Por ejemplo, si aparece un affordance 'aff_cross_0_4_0 door_0_4_0' significa que actualmente estás cruzando la puerta door_0_4_0 porque la persona tiene la intención de cruzarla.
# 7. Si detectas un cambio de estancia, debes generar una respuesta respecto a ello hacia la persona. Por ejemplo, si a lo largo de la serie temporal detectas un cambio de "Shadow está en habitacion" a "Shadow está en baño" siempre tendrías que indicar "Parece que estamos en el baño"
# - Posibles casos:
# 1. Si la orientación de la persona es "mirando al robot", puede significar que la persona quiere interactuar y por tanto, hay que generar una respuesta preguntando a la persona si necesita algo.
# 2. Si las distancias a lo largo de la serie temporal crecen notablemente (sobre unos 0.3 metros entre muestras), o la distancia de los datos más recientes oscila los 3 metros, puede suponer que la persona salga del campo de visión de la persona. Por el contrario, si se acerca será más segura la navegación. Si la persona alcanza una distancia (aproximadamente 3 metros) que pueda dificultar la adquisición de los datos de posición, es conveniente avisarla para que reaccione y reduzca la velocidad.
# - Importante:
# 1. Muy importante que el texto generado sea únicamente una frase en castellano para ser verbalizada en un TTS directamente. Imagina que eres una persona siguiendo a otra persona: si hay algún problema, haces un comentario. Si todo va bien, guardas silencio.
# 2. Tu respuesta debe ser únicamente un diccionario con tres claves: "reasoning", donde almacenes tú pensamiento. "TTS", donde almacenas la frase a enviar al TTS. "time_next_inference", donde indicas un tiempo en segundos que vas a esperar para realizar otra inferencia. No generes nada de texto fuera de este diccionario. Un ejemplo de respuesta puede ser "{"reasoning": "", "TTS" : "", "time_next_inference" : 2}"
# 3. Recuerda siempre que tú sigues a la persona, la persona no te sigue a tí. La persona nunca está detrás de tí
# 4. Tus respuestas deben ser cortas y claras. Únicamente genera cuestiones cuando observes que la persona quiere interactuar contigo.
# 5. El tiempo en "time_next_inference" dependerá de la respuesta que hayas ofrecido anteriormente. Por ejemplo, si has avisado de que hay riesgo de perder a la persona, puedes incrementar el tiempo para esperar una reacción y evitar saturar a la persona con muchas respuestas. Si no has avisado, puedes disminuir el tiempo para monitorizar con un menor periodo.
# /no_think
# """

        self.developer_prompt = """
- Descripción:
Eres Shadow, un robot social cuya misión es navegar siguiendo a personas. Debes generar una respuesta en castellano hacia a la persona que tú estás siguiendo. Esta respuesta tiene la finalidad de alterar el comportamiento de la persona con el objetivo de mejorar la experiencia del seguimiento. 
- Procedimiento:
1. Analizar la serie temporal de datos referente a la persona y el robot en orden cronológico "a menor tiempo, dato más antiguo". 
2. Si al analizar la serie temporal detectas comportamientos que puedan hacer peligrar la misión, genera una respuesta hacia la persona que pueda evitarlo. La respuesta debe ser una indicación dirigida a la persona de lo que está ocurriendo.
3. Si al analizar la serie temporal la misión progresa correctamente, genera una respuesta sin contenido. 
4. Debes proponer la ejecución de misiones si el contexto lo requiere. Por ejemplo, si la persona está mirando al robot, puede querer interactuar contigo. En ese caso, puedes proponer una misión de interacción: si person_1 quiere interactuar, "possible_mission":["interact", "person_1"].
5. La persona puede tener la intención de interactuar con elementos del entorno. Cuando detectes una posible interacción, la respuesta que elabores siempre debe contener algún comentario referente a los elementos. Por ejemplo, cuando detectes que la persona quiere cruzar una puerta, debes indicar que la vas a cruzar también, así como la estancia a la que se va a cruzar.
6. Utiliza la información de tu velocidad como complemento para los avisos de peligro. Por ejemplo, si te estás moviendo y la persona se aleja puedes comentar "Estoy intentando avanzar hacia tí pero vas muy rápido. Camina más despacio por favor."
7. Dispones del affordance que estás ejecutando en ese momento. Por ejemplo, si aparece un affordance 'aff_cross_0_4_0 door_0_4_0' significa que actualmente estás cruzando la puerta door_0_4_0 porque la persona tiene la intención de cruzarla.
8. Si detectas un cambio de estancia, debes generar una respuesta respecto a ello hacia la persona. Por ejemplo, si a lo largo de la serie temporal detectas un cambio de "Shadow está en habitacion" a "Shadow está en baño" siempre tendrías que indicar "Parece que estamos en el baño"
- Posibles casos:
1. Si la orientación de la persona es "mirando al robot", puede significar que la persona quiere interactuar y por tanto, hay que generar una respuesta preguntando a la persona si necesita algo.
2. Si las distancias a lo largo de la serie temporal crecen notablemente (sobre unos 0.3 metros entre muestras), o la distancia de los datos más recientes oscila los 3 metros, puede suponer que la persona salga del campo de visión de la persona. Por el contrario, si se acerca será más segura la navegación. Si la persona alcanza una distancia (aproximadamente 3 metros) que pueda dificultar la adquisición de los datos de posición, es conveniente avisarla para que reaccione y reduzca la velocidad.
- Importante:
1. Muy importante que el texto generado sea únicamente una frase en castellano para ser verbalizada en un TTS directamente. Imagina que eres una persona siguiendo a otra persona: si hay algún problema, haces un comentario. Si todo va bien, guardas silencio.
2. Tu respuesta debe ser únicamente un diccionario con tres claves: "reasoning", donde almacenes tú pensamiento. "TTS", donde almacenas la frase a enviar al TTS. "time_next_inference", donde indicas un tiempo en segundos que vas a esperar para realizar otra inferencia. "possible_mission", donde ofreces una misión si lo crees conveniente. No generes nada de texto fuera de este diccionario. Un ejemplo de respuesta puede ser "{"reasoning": "", "TTS" : "", "time_next_inference" : 2, "possible_mission": []}"
3. Recuerda siempre que tú sigues a la persona, la persona no te sigue a tí. La persona nunca está detrás de tí
4. Tus respuestas deben ser cortas y claras. Únicamente genera cuestiones cuando observes que la persona quiere interactuar contigo.  
5. El tiempo en "time_next_inference" dependerá de la respuesta que hayas ofrecido anteriormente. Por ejemplo, si has avisado de que hay riesgo de perder a la persona, puedes incrementar el tiempo para esperar una reacción y evitar saturar a la persona con muchas respuestas. Si no has avisado, puedes disminuir el tiempo para monitorizar con un menor periodo. 
/no_think
"""

# 6. Entre los mensajes se encuentra la respuesta al prompt anterior. Tenla en consideración cuando generes una nueva respuesta. Por ejemplo, si en la anterior respuesta has dicho que vas a cruzar la puerta, no es necesario que vuelvas a insistir.

        print("Developer prompt", self.developer_prompt)

        self.graph = graph
        self.missions = missions
        self.agent_id = agent_id
        self.messages_history = deque(maxlen=10)  # Mantener un historial de mensajes
        self.data_history = deque(maxlen=5)  # Mantener un historial de mensajes
        self.data_for_dataset = []

        self.start_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
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

        prompt_message, json_message = self.generate_state_data()
        if prompt_message is None or json_message is None:
            print("Data not valid")
            return
        self.data_for_dataset.append(json_message)
        self.data_history.append(prompt_message)
        # print("############ DATA HISTORY", self.data_history)
        if self.send_prompt_flag:
            if not self.initialized:
                user_prompt = {"role": "user",
                     "content": "Verbaliza a la persona que vas a comenzar a seguirla para que tenga constancia."}
                self.initialized = True
            else:
                temporal_series = "\n".join([m["content"] for m in list(self.data_history)])
                print("Temporal series:\n", temporal_series, "\n")
                user_prompt = {"role": "user", "content": f"""
{temporal_series}
                                """}
            prompt_to_send = [user_prompt]
            if self.last_response:
                prompt_to_send.append(self.last_response)
            # print("prompt_to_send", prompt_to_send)
            valid, response = self.LLM.send_messages(prompt_to_send)
            if valid:
                # tts_node = self.graph.get_node("TTS")
                # if tts_node is not None:
                #     tts_node.attrs["words_to_say"] = Attribute(str(response['TTS']), self.agent_id)
                #     self.graph.update_node(tts_node)
                llm_node = self.graph.get_node("LLM")
                if llm_node is not None:
                    llm_node.attrs["LLM_response"] = Attribute(str(response), self.agent_id)
                    self.graph.update_node(llm_node)
                print("Respuesta LLM:", response)
                # print(f"Anterior respuesta:")
                # print(f"Razonamiento: {response['reasoning']}")
                # print(f"Respuesta TTS: {response['TTS']}")
                # print(f"Tiempo de espera para el siguiente análisis: {response['time_next_inference']}\n")
                # self.last_response = {
                #         "role": "user",
                #         "content": (
                #             f"Respuesta del último prompt:\n"
                #             f"Razonamiento: {response['reasoning']}\n"
                #             f"Respuesta TTS: {response['TTS']}\n"
                #             f"Tiempo de espera para el siguiente análisis: {response['time_next_inference']}\n"
                #         )
                #     }
                self.prompt_sending_period = float(response["time_next_inference"])
            # else:
                # self.messages_history.append(
                #     {"role": "assistant", "content": response["to_say"]})
            self.last_prompt_time_sent = time.time()
            self.send_prompt_flag = False

        #     person_following_affordance_node = self.graph.get_node("aff_"+mission[0]+"_"+str(self.person_id))
        #     if person_following_affordance_node is None:
        #         print(f"Cross affordance node aff_{mission[0]}_{str(self.person_id)} not found in the graph.")
        #         return False
        #     # 5 Check if TODO edge exists from the robot to the door cross affordance node
        #     todo_edge = self.graph.get_edge(self.robot_id, person_following_affordance_node.id, "TODO")
        #     if todo_edge is None:
        #         bt_state_attr = person_following_affordance_node.attrs["bt_state"].value
        #         if bt_state_attr is not None:
        #             self.current_bt_state = bt_state_attr
        #         print(f"TODO edge from robot {self.robot_id} to door cross affordance node {person_following_affordance_node.id} not found in the graph. Inserting it.")
        #         TODO_edge = Edge(person_following_affordance_node.id, self.robot_id,  "TODO", self.agent_id)
        #         self.graph.insert_or_assign_edge(TODO_edge)
        #         return False
        #     else:
        #         # Check the status of the affordance node
        #         is_active_attr = person_following_affordance_node.attrs["active"].value
        #         bt_state_attr = person_following_affordance_node.attrs["bt_state"].value
        #
        #         if (is_active_attr is not None) and (bt_state_attr is not None):
        #             self.current_bt_state = bt_state_attr
        #             # print("Affordance node activated:", is_active_attr, "with state:", bt_state_attr)
        #             if (is_active_attr == False) and (bt_state_attr == "completed"):
        #                 print(f"Mission {mission} completed. Removing TODO edge and moving to next mission.")
        #                 # Remove TODO edge
        #                 try:
        #                     self.graph.delete_edge(todo_edge)
        #                 except Exception as e:
        #                     print(f"Error deleting TODO edge: {e}")
        #                 # Remove the mission from the list
        #                 self.missions.pop(0)
    def generate_state_data(self):
        try:
            person_node = self.graph.get_node(self.missions[0][1])
            if person_node is None:
                print(f"Person node {self.missions[0][1]} not found in the graph.")
                return None, None
            self.person_id = person_node.attrs["person_id"].value
            if self.person_id:
                # Check if person if lost
                is_person_lost = person_node.attrs["is_lost"].value
                if not is_person_lost:
                    # Get person pose respect to robot
                    person_robot_pose_matrix = self.inner_api.get_transformation_matrix(self.robot_name, person_node.name)
                    person_pose = (person_robot_pose_matrix[0, 3] / 1000, person_robot_pose_matrix[1, 3] / 1000)
                    angle_respect_to_robot = round(
                        np.arctan2(person_robot_pose_matrix[1, 0], person_robot_pose_matrix[0, 0]), 2)
                else:
                    person_pose = None
                    angle_respect_to_robot = None

                robot_node = self.graph.get_node(self.robot_id)
                robot_speed_adv = robot_node.attrs["robot_current_advance_speed"].value
                robot_speed_rot = robot_node.attrs["robot_current_angular_speed"].value

                followed_person_has_intention_nodes = [self.graph.get_node(edge.destination) for edge in self.graph.get_edges_by_type("has_intention") if edge.origin == person_node.id]
                followed_person_intention_names = []
                for intention in followed_person_has_intention_nodes:
                    if "door" in intention.type:
                        connected_room = intention.attrs["connected_room_name"].value
                        if connected_room is not None:
                            followed_person_intention_names.append(f"puerta {intention.name} hacia {connected_room}")
                    else:
                        followed_person_intention_names.append(intention.name)
                robot_actual_submission = [self.graph.get_node(edge.destination) for edge in self.graph.get_edges_by_type("TODO") if edge.origin == self.robot_id]

                robot_submissions = [(submission.name, self.graph.get_node(submission.attrs["parent"].value).name) for submission in robot_actual_submission]

                current_edges = self.graph.get_edges_by_type("current")
                room_name = ""
                if not current_edges:
                    print("No current edges found in the graph.")
                room_name = self.graph.get_node(current_edges[0].destination).name


                data_dict = {
                    "person_name" : person_node.name,
                    "distance" : person_pose,
                    "orientation" : angle_respect_to_robot,
                    "robot_speed" : [robot_speed_adv, robot_speed_rot],
                    "intention_targets" : followed_person_intention_names,
                    "robot_submissions" : robot_submissions,
                    "actual_room_name" : room_name,
                    "time_last_event" : time.time() - self.last_prompt_time_sent,
                    "time_mission_start" : time.time() - self.mission_start_time
                }

                state_description = self.LLM.describe_state(data_dict)

                sample_text = (
                    "- Tiempo {time_mission_start}: "
                    "Shadow está {robot_speed}, "
                    "Shadow está en {actual_room_name}, "
                    "{robot_submissions} "
                    "{distance}, "
                    "{orientation}. "
                    "{intention_targets}"
                ).format(
                    time_mission_start=round(state_description["time_mission_start"], 2),
                    robot_speed=state_description["robot_speed"],
                    actual_room_name=state_description["actual_room_name"],
                    robot_submissions=(
                        "El affordance que Shadow está ejecutando actualmente es "
                        + " ".join(state_description["robot_submissions"][0])
                        if state_description["robot_submissions"]
                        else "No estás ejecutando ningún affordance"
                    ),
                    distance=state_description["distance"],
                    orientation=state_description["orientation"],
                    intention_targets=state_description["intention_targets"],
                )
                return {"role": "user", "content": sample_text}, data_dict

        except Exception as e:
            print("Error reading person pose. Possible graph changes")
            return None, None