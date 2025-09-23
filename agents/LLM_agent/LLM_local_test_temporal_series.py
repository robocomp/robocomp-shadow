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
import re
from datetime import datetime
from collections import deque
import math
import copy

LMSTUDIO_API_URL = "http://192.168.50.37:3000/v1"
MODEL_NAME = ""
LMS_PATH = "/home/robolab/.lmstudio/bin/lms"

server_proc = None
modelo_proc = None
appimage_proc = None

# Variable global para guardar el historial de mensajes
messages = []
expended_times = []

memory_limit = 5
messages_memory = deque(maxlen=memory_limit)

model_name = "openai/gpt-oss-20b"

last_answer = None

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
    global messages, expended_times, last_answer
    client = OpenAI(base_url=LMSTUDIO_API_URL, api_key="lm-studio")

    main_prompt = """
- Descripción:
Eres Shadow, un robot social cuya misión es navegar siguiendo a personas. Debes generar una respuesta en castellano hacia a la persona que tú estás siguiendo. Esta respuesta tiene la finalidad de alterar el comportamiento de la persona con el objetivo de mejorar la experiencia del seguimiento. 
- Procedimiento:
1. Analizar la serie temporal de datos referente a la persona y el robot en orden cronológico "a menor tiempo, dato más antiguo". 
2. Si al analizar la serie temporal detectas comportamientos que puedan hacer peligrar la misión, genera una respuesta hacia la persona que pueda evitarlo. La respuesta debe ser una indicación dirigida a la persona de lo que está ocurriendo.
3. Si al analizar la serie temporal la misión progresa correctamente, genera una respuesta sin contenido. 
4. La persona puede tener la intención de interactuar con elementos del entorno. Cuando detectes una posible interacción, la respuesta que elabores siempre debe contener algún comentario referente a los elementos. Por ejemplo, cuando detectes que la persona quiere cruzar una puerta, debes indicar que la vas a cruzar también, así como la estancia a la que se va a cruzar.
5. Utiliza la información de tu velocidad como complemento para los avisos de peligro. Por ejemplo, si te estás moviendo y la persona se aleja puedes comentar "Estoy intentando avanzar hacia tí pero vas muy rápido. Camina más despacio por favor."
6. Dispones del affordance que estás ejecutando en ese momento. Por ejemplo, si aparece un affordance 'aff_cross_0_4_0 door_0_4_0' significa que actualmente estás cruzando la puerta door_0_4_0 porque la persona tiene la intención de cruzarla.
7. Si detectas un cambio de estancia, debes generar una respuesta respecto a ello hacia la persona. Por ejemplo, si a lo largo de la serie temporal detectas un cambio de "Shadow está en habitacion" a "Shadow está en baño" siempre tendrías que indicar "Parece que estamos en el baño"
- Posibles casos:
1. Si la orientación de la persona es "mirando al robot", puede significar que la persona quiere interactuar y por tanto, hay que generar una respuesta preguntando a la persona si necesita algo.
2. Si las distancias a lo largo de la serie temporal crecen notablemente (sobre unos 0.3 metros entre muestras), o la distancia de los datos más recientes oscila los 3 metros, puede suponer que la persona salga del campo de visión de la persona. Por el contrario, si se acerca será más segura la navegación. Si la persona alcanza una distancia (aproximadamente 3 metros) que pueda dificultar la adquisición de los datos de posición, es conveniente avisarla para que reaccione y reduzca la velocidad.
- Importante:
1. Muy importante que el texto generado sea únicamente una frase en castellano para ser verbalizada en un TTS directamente. Imagina que eres una persona siguiendo a otra persona: si hay algún problema, haces un comentario. Si todo va bien, guardas silencio.
2. Tu respuesta debe ser únicamente un diccionario con tres claves: "reasoning", donde almacenes tú pensamiento. "TTS", donde almacenas la frase a enviar al TTS. "time_next_inference", donde indicas un tiempo en segundos que vas a esperar para realizar otra inferencia. No generes nada de texto fuera de este diccionario. Un ejemplo de respuesta puede ser "{"reasoning": "", "TTS" : "", "time_next_inference" : 2}"
3. Recuerda siempre que tú sigues a la persona, la persona no te sigue a tí. La persona nunca está detrás de tí
4. Tus respuestas deben ser cortas y claras. Únicamente genera cuestiones cuando observes que la persona quiere interactuar contigo.  
5. El tiempo en "time_next_inference" dependerá de la respuesta que hayas ofrecido anteriormente. Por ejemplo, si has avisado de que hay riesgo de perder a la persona, puedes incrementar el tiempo para esperar una reacción y evitar saturar a la persona con muchas respuestas. Si no has avisado, puedes disminuir el tiempo para monitorizar con un menor periodo. 
/no_think
"""
# Ejemplos con datos:
# ------------------------------------
# Prompt:
# Tiempo=59.0, distancia frontal=3.05 metros delante del robot, distancia lateral=0.11 metros derecha, orientación=en el mismo sentido que el robot
# Tiempo=60.5, distancia frontal=2.63 metros delante del robot, distancia lateral=0.16 metros derecha, orientación=en el mismo sentido que el robot
# Tiempo=62.0, distancia frontal=2.06 metros delante del robot, distancia lateral=0.12 metros derecha, orientación=en el mismo sentido que el robot
# Tiempo=63.5, distancia frontal=1.88 metros delante del robot, distancia lateral=0.47 metros izquierda, orientación=hacia la izquierda
# Tiempo=65.0, distancia frontal=1.95 metros delante del robot, distancia lateral=0.12 metros derecha, orientación=en el mismo sentido que el robot
# Tiempo=67.0, distancia frontal=1.75 metros delante del robot, distancia lateral=0.04 metros derecha, orientación=en el mismo sentido que el robot
# Tiempo=68.5, distancia frontal=1.14 metros delante del robot, distancia lateral=0.01 metros derecha, orientación=en el mismo sentido que el robot
# Tiempo=70.0, distancia frontal=1.08 metros delante del robot, distancia lateral=0.00 metros izquierda, orientación=en el mismo sentido que el robot
# Tiempo=71.5, distancia frontal=1.07 metros delante del robot, distancia lateral=0.01 metros derecha, orientación=en el mismo sentido que el robot
#
# Respuesta:
#
# ------------------------------------
# Prompt:
# Tiempo=42.0, distancia frontal=0.74 metros delante del robot, distancia lateral=0.01 metros derecha, orientación=hacia la izquierda
# Tiempo=43.5, distancia frontal=1.27 metros delante del robot, distancia lateral=0.16 metros izquierda, orientación=en el mismo sentido que el robot
# Tiempo=45.0, distancia frontal=1.85 metros delante del robot, distancia lateral=0.36 metros izquierda, orientación=hacia la izquierda
# Tiempo=46.5, distancia frontal=2.42 metros delante del robot, distancia lateral=0.50 metros izquierda, orientación=hacia la izquierda
# Tiempo=48.0, distancia frontal=3.11 metros delante del robot, distancia lateral=0.54 metros izquierda, orientación=en el mismo sentido que el robot
# Tiempo=49.5, distancia frontal=3.28 metros delante del robot, distancia lateral=0.57 metros izquierda, orientación=hacia la izquierda
#
# Respuesta:
# Te estás alejando notablemente.
# """
    messages = [{
        "role": "system",
        "content": main_prompt
    }]

    expended_times = []
    if mensajes_json:
        print("📄 Ejecutando conversación desde JSON...\n")
        for idx, msg in enumerate(mensajes_json):
            data_dict = describe_state(msg)
            if 'intentions' in msg:
                sample_text = f"Tiempo={round(msg['time_mission_start'] - mensajes_json[0]['time_mission_start'], 2)}, distancia frontal={data_dict['frontal_distance']}, distancia lateral={data_dict['lateral_distance']}, orientación={data_dict['orientation']}, intenciones={msg['intentions']}"
            else:
                sample_text = f"Tiempo={round(msg['time_mission_start'] - mensajes_json[0]['time_mission_start'], 2)}, distancia frontal={data_dict['frontal_distance']}, distancia lateral={data_dict['lateral_distance']}, orientación={data_dict['orientation']}"
            # Añadimos la nueva muestra a la cola circular
            messages_memory.append({"role": "user", "content": sample_text})

            # Construimos la serie temporal concatenando las muestras en orden
            temporal_series = "\n".join([m["content"] for m in messages_memory])

            # Creamos un solo prompt con la serie temporal completa
            developer_prompt = f"""
{temporal_series}
                """
            print("Developer prompt", developer_prompt)
            # Agregamos a messages el prompt único con toda la ventana temporal
            messages = [{"role": "system", "content": main_prompt}] + [{"role": "user", "content": developer_prompt}]
            if last_answer != None:
                messages += [last_answer]

            #     print("Messages", message)
            try:
                start = time.time()
                if memory_limit > 0:
                    print("Memory size", len(messages_memory))
                    completion = client.chat.completions.create(
                        model=modelo_seleccionado,
                        messages=messages,
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
                match = re.search(r'\{.*\}', reply)
                if match:
                    json_str = match.group(0)
                    diccionario = json.loads(json_str)
                    print(diccionario)
                else:
                    print("No se encontró JSON en el texto.")
                # print(f"[{idx+1}/{len(mensajes_json)}] 🤖 Respuesta: {reply}\n")
                # messages.append({"role": "assistant", "content": reply})
                # messages_memory.append({"role": "assistant", "content": reply})
                last_answer = {"role": "assistant", "content": reply}
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
                direction = "izquierda" if x < 0 else ("derecha" if x > 0 else "centrado")
                result["lateral_distance"] = f"{abs(x):.2f} metros {direction}"
            if y != 0:
                direction = "detrás" if y < 0 else "delante"
                result["frontal_distance"] = f"{abs(y):.2f} metros {direction} del robot"

    # Descripción de orientación: float (radianes) → texto
    if "orientation" in state:
        angle = state["orientation"]
        angle = (angle + math.pi) % (2 * math.pi) - math.pi  # normaliza a [-π, π]
        if abs(angle) < math.pi / 6:
            result["orientation"] = "en el mismo sentido que el robot"
        elif abs(angle) > 5 * math.pi / 6:
            result["orientation"] = "mirando al robot"
        elif angle > 0:
            result["orientation"] = f"hacia la izquierda {round(angle,1)} radianes"
        else:
            result["orientation"] = f"hacia la derecha {round(angle,1)} radianes"

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
        # modelo_cargado = obtener_modelo_lanzado_lmstudio()
        # model_name = modelo_cargado
        if model_name is not None:
            try:
                chat_local(model_name, mensajes_json)
            except KeyboardInterrupt:
                manejar_interrupcion(None, None)
    else:
        print("⚠️ No se pudo verificar la disponibilidad del servidor.")
