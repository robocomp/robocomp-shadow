import pyttsx3

class TTS_Pyttsx3:
    def __init__(self):
        # Inicializa el motor TTS
        self.engine = pyttsx3.init()
        # Opcional: cambiar la velocidad y volumen
        self.engine.setProperty("rate", 120)  # Velocidad de la voz
        self.engine.setProperty("volume", 1)  # Volumen (0.0 a 1.0)

    def say(self, text):
        # Decir el texto
        self.engine.say(text)
        self.engine.runAndWait()