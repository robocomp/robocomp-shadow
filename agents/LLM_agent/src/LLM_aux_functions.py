import copy

import numpy as np
from openai import OpenAI
from collections import deque
import time
import re
import tiktoken
import json
import math

class LLMFunctions:
    def __init__(self, dev_prompt):
        self.developer_prompt = [{"role": "system", "content": dev_prompt}]
        self.model = "openai/gpt-oss-20b"
        # self.client = OpenAI(base_url="http://192.168.50.37:3000/v1", api_key="lm-studio")
        self.client = OpenAI(base_url="http://158.49.112.181:8080/v1", api_key="lm-studio")

    def send_messages(self, messages):
        try:
            messages_to_send = self.developer_prompt + messages
            # print("Messages to send:", messages_to_send)
            # print("🔢 Tokens usados:", self.contar_tokens(messages_to_send))
            start = time.time()
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=messages_to_send,
                temperature=0.5
            )
            # print("########## Response", completion)
            # print("Expended time:", time.time() - start)
            match = re.search(r'\{[\s\S]*\}', completion.choices[0].message.content.strip(), re.DOTALL)
            if match:
                json_str = match.group()
                diccionario = json.loads(json_str)
                return True, diccionario
            else:
                print("No se encontró JSON en el texto.")
                return False, None


        except Exception as e:
            return False, None

    def describe_state(self, state: dict) -> dict:
        result = copy.copy(state)

        # Descripción de distancia: [x, y] → "X meters left/right, Y meters front/back"
        if "distance" in state:
            values = state["distance"]
            if isinstance(values, (list, tuple)) and len(values) == 2:
                x, y = values
                direction_lateral = "izquierda" if x < 0 else ("derecha" if x > 0 else "centrado")
                result_lateral = f"{abs(x):.2f} metros {direction_lateral}"
                direction_frontal = "detrás" if y < 0 else "delante"
                result_lineal = f"{abs(y):.2f} metros {direction_frontal} del robot"
                result["distance"] = f"La persona está a una distancia de {result_lateral} y {result_lineal}"
            else:
                result["distance"] = f"La persona está fuera del campo de visión del robot"

        # Descripción de orientación: float (radianes) → texto
        if "orientation" in state:
            angle = state["orientation"]
            if isinstance(angle, (int, float)):
                angle = (angle + math.pi) % (2 * math.pi) - math.pi  # normaliza a [-π, π]
                print(f"Orientation angle: {angle}")
                if abs(angle) < math.pi / 6:
                    result["orientation"] = "orientada en el sentido opuesto al robot"
                elif abs(angle) > 5 * math.pi / 6:
                    result["orientation"] = "orientada hacia el robot"
                elif angle > 0:
                    result["orientation"] = f"orientada hacia la izquierda {round(angle, 1)} radianes"
                else:
                    result["orientation"] = f"orientada hacia la derecha {round(angle, 1)} radianes"
            else:
                result["orientation"] = ""


        # Descripción de velocidad del robot: [lineal, angular]
        if "robot_speed" in state:
            speed = state["robot_speed"]
            if isinstance(speed, (list, tuple)) and len(speed) == 2:
                lin, ang = speed
                desc = []
                if abs(lin) < 0.05 and abs(ang) < 0.05:
                    desc.append("detenido")
                else:
                    if abs(lin) >= 0.05:
                        direction = "hacia delante" if lin > 0 else "hacia atrás"
                        desc.append(f"moviéndose {direction} a {abs(lin):.2f} m/s")
                    if abs(ang) >= 0.05:
                        direction = "izquierda" if ang > 0 else "derecha"
                        desc.append(f"rotando hacia la {direction} a {abs(ang):.2f} rad/s")
                result["robot_speed"] = ", ".join(desc)

        if "intention_targets" in state:
            intentions = state["intention_targets"]
            if intentions:
                result["intention_targets"] = 'Es posible que la persona quiera interactuar con ' + ','.join(intentions)
            else:
                result["intention_targets"] = 'La persona no tiene intenciones de interacción'
        return result
