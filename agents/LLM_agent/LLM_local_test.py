import subprocess
import requests
import socket
import os
import time
import signal
import sys
import psutil
from openai import OpenAI
import json
from datetime import datetime
from collections import deque
import math
import copy

LMSTUDIO_API_URL = "http://localhost:3000/v1/models"
MODEL_NAME = ""
LMS_PATH = "/home/robolab/.lmstudio/bin/lms"

server_proc = None
modelo_proc = None
appimage_proc = None

# Variable global para guardar el historial de mensajes
messages = []
expended_times = []

memory_limit = 12
messages_memory = deque(maxlen=memory_limit)

model_name = ""

# ------------------- NUEVO -------------------
def listar_json_dataset():
    dataset_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset")
    if not os.path.exists(dataset_dir):
        print(f"❌ No existe la carpeta: {dataset_dir}")
        return []
    archivos_json = [f for f in os.listdir(dataset_dir) if f.endswith(".json")]
    return [os.path.join(dataset_dir, f) for f in archivos_json]

def seleccionar_json():
    archivos = listar_json_dataset()
    if not archivos:
        print("⚠️ No se encontraron archivos .json en la carpeta 'dataset'.")
        return None
    print("\n📂 Archivos JSON disponibles en 'dataset':")
    for i, archivo in enumerate(archivos):
        print(f"{i+1}. {os.path.basename(archivo)}")
    while True:
        try:
            eleccion = int(input("\nElige un archivo (número, 0 para cancelar): "))
            if eleccion == 0:
                return None
            if 1 <= eleccion <= len(archivos):
                return archivos[eleccion - 1]
            else:
                print("❗ Opción fuera de rango.")
        except ValueError:
            print("❗ Introduce un número válido.")

def cargar_json(ruta_json):
    try:
        with open(ruta_json, "r", encoding="utf-8") as f:
            data = json.load(f)
        if "messages" not in data or not isinstance(data["messages"], list):
            raise ValueError("El JSON no contiene la clave 'messages' con una lista.")
        return data["messages"]
    except Exception as e:
        print(f"❌ Error al leer el JSON: {e}")
        return []

# Guardar historial en un JSON
def guardar_historial():
    if not messages:
        return
    carpeta_historial = os.path.join(os.path.dirname(os.path.abspath(__file__)), "historial")
    os.makedirs(carpeta_historial, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    ruta = os.path.join(
        carpeta_historial,
        f"historial_{timestamp}_memory_size_{memory_limit}_{model_name.replace('/', '-')}.json"
    )
    try:
        for i, expended_time in enumerate(expended_times):
            messages[i*2+1]["LLM_expended_time"] = expended_time
        with open(ruta, "w", encoding="utf-8") as f:
            json.dump({"messages": messages}, f, ensure_ascii=False, indent=2)
        print(f"💾 Historial guardado en: {ruta}")
    except Exception as e:
        print(f"❌ Error al guardar historial: {e}")

# Handler para Ctrl+C
def manejar_interrupcion(sig, frame):
    print("\n⚡ Interrumpido por usuario. Guardando historial...")
    guardar_historial()
    sys.exit(0)

signal.signal(signal.SIGINT, manejar_interrupcion)

def esperar_api(timeout=60):
    print("⏳ Esperando a que la API de LM Studio esté disponible...")
    inicio = time.time()
    while time.time() - inicio < timeout:
        try:
            r = requests.get(LMSTUDIO_API_URL)
            if r.status_code == 200:
                print("✅ LM Studio API está activa.")
                return True
        except requests.exceptions.ConnectionError:
            pass
        time.sleep(1)
    print("❌ Timeout: la API de LM Studio no respondió.")
    return False

def obtener_modelo_lanzado_lmstudio():
    try:
        resultado = subprocess.run(
            [LMS_PATH, "ps", "--json"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
            text=True
        )
        modelos = json.loads(resultado.stdout)
        if len(modelos) > 0:
            return modelos[0]["identifier"]
    except subprocess.CalledProcessError as e:
        print("❌ Error al obtener la lista de modelos:", e.stderr)
        return None


def chat_local(modelo_seleccionado, mensajes_json=None):
    global messages, expended_times
    client = OpenAI(base_url="http://localhost:3000/v1", api_key="lm-studio")

    main_prompt = """
Descripción:
Eres un robot social diseñado cuya misión navegar siguiendo a personas. Debes generar una respuesta hacia a la persona que estás siguiendo. Esta respuesta tiene la finalidad de alterar el comportamiento de la persona con el objetivo de mejorar la experiencia del seguimiento. Tus respuestas deben ser cortas y claras.
Procedimiento:
Debes:
1. Analizar toda la información referente a la persona y el robot. 
2. Si al analizar la información la misión peligra, genera una respuesta que infieras que puede mejorar la misión.
3. Si al analizar la información la misión progresa correctamente, NO generes una respuesta. 
Ejemplos *los números entre paréntesis son rangos aceptados para cada caso*:
***
- Información: 'La persona se encuentra respecto al robot (0.5 - 2.0) metros delante y (-0.4, 0.4) metros derecha. La orientación de la persona relativa al robot es en el mismo sentido que el robot.'
- Respuesta: ''
***
- Información: 'La persona se encuentra respecto al robot (< 2.0) metros delante y (-1.5, 1.5) metros derecha. La orientación de la persona relativa al robot es en el mismo sentido que el robot.'
- Respuesta: 'Te estás alejando demasiado.'
***
- Información: 'La persona se encuentra respecto al robot (1.0 - 2.0) metros delante y (-0.4, 0.4) metros derecha. La orientación de la persona relativa al robot es mirando al robot.'
- Respuesta: '¿Puedo ayudarte en algo?'
"""

    messages = [{
        "role": "system",
        "content": main_prompt
    }]

    expended_times = []
    if mensajes_json:
        print("📄 Ejecutando conversación desde JSON...\n")
        for idx, msg in enumerate(mensajes_json):
            data_dict = describe_state(msg)
            developer_prompt = """
La posición de la persona respecto al robot es {frontal_distance} y {lateral_distance}.
La orientación de la persona relativa al robot es {orientation}.
            """

            messages.append({"role": "user", "content": developer_prompt.format(
                frontal_distance=data_dict["frontal_distance"],
                lateral_distance=data_dict["lateral_distance"],
                orientation=data_dict["orientation"]
            )})
            messages_memory.append({"role": "user", "content": developer_prompt.format(
                frontal_distance=data_dict["frontal_distance"],
                lateral_distance=data_dict["lateral_distance"],
                orientation=data_dict["orientation"]
            )})
            print(developer_prompt.format(
                frontal_distance=data_dict["frontal_distance"],
                lateral_distance=data_dict["lateral_distance"],
                orientation=data_dict["orientation"]
            ))
            if len(messages_memory) == memory_limit:
                messages_memory.popleft()

            aux_message_list = [{"role": "system","content": main_prompt}] + list(messages_memory)
            # for message in aux_message_list:
            #     print("Messages", message)
            try:
                start = time.time()
                if memory_limit > 0:
                    print("Memory size", len(messages_memory))
                    completion = client.chat.completions.create(
                        model=modelo_seleccionado,
                        messages=aux_message_list,
                        temperature=0.5,
                    )
                else:
                    completion = client.chat.completions.create(
                        model=modelo_seleccionado,
                        messages=messages,
                        temperature=0.5,
                    )
                expended_milliseconds = time.time() - start
                expended_times.append(expended_milliseconds)
                reply = completion.choices[0].message.content
                print(f"[{idx+1}/{len(mensajes_json)}] 🤖 Respuesta: {reply}\n")
                messages.append({"role": "assistant", "content": reply})
                messages_memory.append({"role": "assistant", "content": reply})
            except Exception as e:
                print("❌ Error al generar respuesta:", e)
                break
        guardar_historial()
    else:
        print("🤖 LLM Chat (escribe 'salir' para terminar)\n")
        while True:
            user_input = input("🧑 Tú: ")
            if user_input.lower() in {"salir", "exit", "quit"}:
                print("👋 Hasta luego.")
                guardar_historial()
                break
            messages.append({"role": "user", "content": user_input})
            try:
                completion = client.chat.completions.create(
                    model=modelo_seleccionado,
                    messages=messages,
                    # temperature=0.7,
                )
                reply = completion.choices[0].message.content
                print(f"🤖 Asistente: {reply}\n")
                messages.append({"role": "assistant", "content": reply})
            except Exception as e:
                print("❌ Error al generar respuesta:", e)
                break

def describe_state(state: dict) -> dict:
    result = copy.copy(state)

    # Descripción de distancia: [x, y] → "X meters left/right, Y meters front/back"
    if "distance" in state:
        values = state["distance"]
        if isinstance(values, (list, tuple)) and len(values) == 2:
            desc = []
            x, y = values
            if x != 0:
                direction = "izquierda" if x < 0 else "derecha"
                result["lateral_distance"] = f"{abs(x):.1f} metros {direction}"
            if y != 0:
                direction = "atrás" if y < 0 else "delante"
                result["frontal_distance"] = f"{abs(y):.1f} metros {direction}"

    # Descripción de orientación: float (radianes) → texto
    if "orientation" in state:
        angle = state["orientation"]
        angle = (angle + math.pi) % (2 * math.pi) - math.pi  # normaliza a [-π, π]
        if abs(angle) < math.pi / 6:
            result["orientation"] = "en el mismo sentido que el robot"
        elif abs(angle) > 5 * math.pi / 6:
            result["orientation"] = "mirando el robot"
        elif angle > 0:
            result["orientation"] = "hacia la izquierda"
        else:
            result["orientation"] = "hacia la derecha"

    # Descripción de velocidad del robot: [lineal, angular]
    if "robot_speed" in state:
        speed = state["robot_speed"]
        if isinstance(speed, (list, tuple)) and len(speed) == 2:
            lin, ang = speed
            desc = []
            if abs(lin) < 0.05 and abs(ang) < 0.05:
                desc.append("robot is stopped")
            else:
                if abs(lin) >= 0.05:
                    direction = "forward" if lin > 0 else "backward"
                    desc.append(f"moving {direction} at {abs(lin):.2f} m/s")
                if abs(ang) >= 0.05:
                    direction = "left" if ang > 0 else "right"
                    desc.append(f"rotating {direction} at {abs(ang):.2f} rad/s")
            result["robot_speed"] = ", ".join(desc)

    return result

if __name__ == "__main__":
    if esperar_api():
        ruta_json = seleccionar_json()
        mensajes_json = cargar_json(ruta_json) if ruta_json else None
        modelo_cargado = obtener_modelo_lanzado_lmstudio()
        model_name = modelo_cargado
        if modelo_cargado is not None:
            try:
                chat_local(modelo_cargado, mensajes_json)
            except KeyboardInterrupt:
                manejar_interrupcion(None, None)
    else:
        print("⚠️ No se pudo verificar la disponibilidad del servidor.")
