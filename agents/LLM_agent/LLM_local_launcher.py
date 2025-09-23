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

LMSTUDIO_API_URL = "http://localhost:443/v1/models"
MODEL_NAME = ""
LMS_PATH = "/home/robolab/.lmstudio/bin/lms"
LM_STUDIO_APPIMAGE = "/home/robolab/Gerardo/LM-Studio-0.3.22-2-x64.AppImage"

server_proc = None
modelo_proc = None
appimage_proc = None

# ------------------- NUEVO -------------------
def listar_json_dataset():
    """Lista los archivos .json dentro de la carpeta 'dataset' en el mismo directorio del script."""
    dataset_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset")
    if not os.path.exists(dataset_dir):
        print(f"❌ No existe la carpeta: {dataset_dir}")
        return []

    archivos_json = [f for f in os.listdir(dataset_dir) if f.endswith(".json")]
    return [os.path.join(dataset_dir, f) for f in archivos_json]

def seleccionar_json():
    """Permite seleccionar un archivo JSON de la carpeta dataset."""
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
    """Carga el archivo JSON que contiene 'messages' y devuelve la lista."""
    try:
        with open(ruta_json, "r", encoding="utf-8") as f:
            data = json.load(f)
        if "messages" not in data or not isinstance(data["messages"], list):
            raise ValueError("El JSON no contiene la clave 'messages' con una lista.")
        return data["messages"]
    except Exception as e:
        print(f"❌ Error al leer el JSON: {e}")
        return []
# ----------------------------------------------

def iniciar_lm_studio():
    global appimage_proc
    print("🚀 Lanzando LM Studio AppImage...")
    appimage_proc = subprocess.Popen([LM_STUDIO_APPIMAGE], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(5)

def lanzar_lmstudio_server():
    global server_proc
    print("🚀 Iniciando LM Studio server...")
    server_proc = subprocess.run([LMS_PATH, "server", "start"], check=True)

def cerrar_lmstudio_server():
    global server_proc
    try:
        subprocess.run([LMS_PATH, "server", "stop"], check=True)
    except:
        print("❌ Error al detener el servidor de LM Studio.")

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
    llms = [m for m in modelos_disponibles if m.get("type") == "llm"]
    if not llms:
        print("❌ No hay modelos LLM disponibles.")
        return None

    print("\n📦 Modelos LLM disponibles:\n")
    for i, modelo in enumerate(llms):
        print(f"{i + 1}. {modelo['displayName']} (params: {modelo['paramsString']}, arquitectura: {modelo['architecture']})")

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
    print(f"📦 Cargando modelo '{modelo_seleccionado}'...")
    subprocess.run([LMS_PATH, "load", modelo_seleccionado], check=True)
    time.sleep(3)

def borrar_modelo():
    try:
        subprocess.run([LMS_PATH, "unload", "--all"], check=True)
        time.sleep(3)
    except:
        print("❌ Error al borrar el modelo.")

def cerrar_procesos():
    print("\n🛑 Cerrando procesos...")
    borrar_modelo()
    cerrar_lmstudio_server()
    matar_lmstudio_en_tmp()

def matar_lmstudio_en_tmp():
    subprocess.Popen(["pkill", "-f", "lm-studio"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def signal_handler(sig, frame):
    print(f"\n⚡ Señal {sig} recibida, cerrando...")
    cerrar_procesos()
    sys.exit(0)

if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    iniciar_lm_studio()
    lanzar_lmstudio_server()

    if esperar_api():
        modelos = obtener_modelos_lmstudio()
        modelo_seleccionado = elegir_llm(modelos)
        cargar_modelo(modelo_seleccionado)

        try:
            while esperar_api():
                time.sleep(5)
                pass
            cerrar_procesos()
        except KeyboardInterrupt:
            print("\n⚡ Interrumpido por usuario.")
        finally:
            cerrar_procesos()
    else:
        print("⚠️ No se pudo verificar la disponibilidad del servidor.")
        cerrar_procesos()
