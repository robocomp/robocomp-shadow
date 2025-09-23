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

LMSTUDIO_API_URL = "http://192.168.50.19:3000/v1/models"
MODEL_NAME = ""
LMS_PATH = "/home/robolab/.lmstudio/bin/lms"  # Cambia aquí por la ruta absoluta a 'lms'

server_proc = None
modelo_proc = None
appimage_proc = None

LM_STUDIO_APPIMAGE = "/home/robolab/robocomp/components/robocomp-shadow/agents/LLM_agent/LM-Studio-0.3.20-4-x64.AppImage"  # o la AppImage con su ruta completa

def iniciar_lm_studio():
    global appimage_proc
    print("🚀 Lanzando LM Studio AppImage...")
    appimage_proc = subprocess.Popen([LM_STUDIO_APPIMAGE], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(5)  # Deja que se inicie el GUI, aunque no interactuemos


def lanzar_lmstudio_server():
    global server_proc
    print("🚀 Iniciando LM Studio server con 'lms server start'...")
    server_proc = subprocess.run(
        [LMS_PATH, "server", "start"],
            check = True
            # stdout=subprocess.DEVNULL,
            # stderr=subprocess.DEVNULL
        )

def cerrar_lmstudio_server():
    global server_proc
    print("🚀 Iniciando LM Studio server con 'lms server start'...")
    try:
        server_proc = subprocess.run(
            [LMS_PATH, "server", "stop"],
            check=True
            # stdout=subprocess.DEVNULL,
            # stderr=subprocess.DEVNULL
        )
    except:
        print("❌ Error al detener el servidor de LM Studio. Puede que no esté activo.")
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


def obtener_modelos_lmstudio():
    """Ejecuta `lms ls --json` y devuelve la lista de modelos como diccionarios."""
    try:
        resultado = subprocess.run(
            [LMS_PATH, "ls", "--json"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
            text=True
        )
        modelos = json.loads(resultado.stdout)
        return modelos
    except subprocess.CalledProcessError as e:
        print("❌ Error al obtener la lista de modelos:", e.stderr)
        return []


def elegir_llm(modelos_disponibles):
    """
    Permite al usuario seleccionar un modelo LLM de la lista dada.
    """
    llms = [m for m in modelos_disponibles if m.get("type") == "llm"]

    if not llms:
        print("❌ No hay modelos LLM disponibles.")
        return None

    print("\n📦 Modelos LLM disponibles:\n")
    for i, modelo in enumerate(llms):
        print(
            f"{i + 1}. {modelo['displayName']} (params: {modelo['paramsString']}, arquitectura: {modelo['architecture']})")

    while True:
        try:
            eleccion = int(input("\nElige un modelo (número): ")) - 1
            if 0 <= eleccion < len(llms):
                return llms[eleccion]["modelKey"]
            else:
                print("❗ Opción fuera de rango.")
        except ValueError:
            print("❗ Introduce un número válido.")


def cargar_modelo(modelo_seleccionado):
    global modelo_proc
    print(f"📦 Cargando modelo '{modelo_seleccionado}'...")
    modelo_proc = subprocess.run(
        [LMS_PATH, "load", modelo_seleccionado],
         check = True
        # stdout=subprocess.DEVNULL,
        # stderr=subprocess.DEVNULL
    )
    time.sleep(3)  # Esperar un poco para que el modelo arranque

def borrar_modelo():
    global modelo_proc
    try:
        modelo_proc = subprocess.run(
            [LMS_PATH, "unload", "--all"],
             check = True
            # stdout=subprocess.DEVNULL,
            # stderr=subprocess.DEVNULL
        )
        time.sleep(3)  # Esperar un poco para que el modelo arranque
    except:
        print("❌ Error al borrar el modelo. Puede que no haya ninguno cargado.")

def cerrar_procesos():
    print("\n🛑 Cerrando procesos...")
    borrar_modelo()
    cerrar_lmstudio_server()
    matar_lmstudio_en_tmp()


def matar_lmstudio_en_tmp():
    print("🔍 Buscando procesos de LM Studio en /tmp/.mount_...")
    appimage_proc = subprocess.Popen(["pkill", "-f", "lm-studio"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def signal_handler(sig, frame):
    print(f"\n⚡ Señal {sig} recibida, cerrando...")
    cerrar_procesos()
    sys.exit(0)

def chat_local():
    # Conectar al servidor local de LM Studio
    client = OpenAI(base_url="http://192.168.50.19:3000/v1", api_key="lm-studio")

    # Historial de mensajes del chat
    messages = []
    messages.append({"role": "user", "content": "No utilices el modo razonamiento."})

    print("🤖 LLM Chat (escribe 'salir' para terminar)\n")

    while True:
        user_input = input("🧑 Tú: ")
        if user_input.lower() in {"salir", "exit", "quit"}:
            print("👋 Hasta luego.")
            break

        # Añadir mensaje del usuario al historial
        messages.append({"role": "user", "content": user_input})

        try:
            start = time.time()
            # Llamar al modelo local
            completion = client.chat.completions.create(
                model="google/gemma-3-12b",
                messages=messages,
                temperature=0.7,
            )
            print("Expended time:", time.time())

            # Obtener y mostrar respuesta
            reply = completion.choices[0].message.content
            print(f"🤖 Asistente: {reply}\n")

            # Añadir respuesta al historial
            messages.append({"role": "assistant", "content": reply})

        except Exception as e:
            print("❌ Error al generar respuesta:", e)
            break

if __name__ == "__main__":
    if esperar_api():
        print("🎯 Todo listo. Presiona Ctrl+C para salir.")
        try:
            chat_local()
        except KeyboardInterrupt:
            print("\n⚡ Interrumpido por usuario.")

    else:
        print("⚠️ No se pudo verificar la disponibilidad del servidor.")
