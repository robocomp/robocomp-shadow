import json
import tkinter as tk
from tkinter import simpledialog, messagebox

FILE = "src/main_prompts.json"

def cargar_prompts():
    with open(FILE, "r", encoding="utf-8") as f:
        return json.load(f)

def guardar_prompts(data):
    with open(FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

def editar_prompt(clave):
    nuevo_texto = simpledialog.askstring("Editar prompt", f"Nuevo texto para '{clave}':", initialvalue=data[clave])
    if nuevo_texto is not None:
        data[clave] = nuevo_texto
        guardar_prompts(data)
        messagebox.showinfo("Guardado", f"Prompt '{clave}' actualizado.")

data = cargar_prompts()

root = tk.Tk()
root.title("Editor de Prompts")

for clave in data:
    btn = tk.Button(root, text=clave, command=lambda c=clave: editar_prompt(c))
    btn.pack(fill="x")

root.mainloop()
