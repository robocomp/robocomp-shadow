#!/usr/bin/python3
# -*- coding: utf-8 -*-
#
#    Copyright (C) 2025 by YOUR NAME HERE
#
#    This file is part of RoboComp
#
#    RoboComp is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    RoboComp is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with RoboComp.  If not, see <http://www.gnu.org/licenses/>.
#

from PySide2.QtCore import QTimer
from PySide2.QtWidgets import QApplication
from rich.console import Console
from tensorflow.python.ops.gen_functional_ops import device_index

from genericworker import *
import interfaces as ifaces

import pyaudio
import wave
import numpy as np
import whisper
import time
from collections import deque
from pydub import AudioSegment

# NOISE REDUCTION
import librosa
import noisereduce as nr
import soundfile as sf

import openwakeword
from openwakeword.model import Model

sys.path.append('/opt/robocomp/lib')
console = Console(highlight=False)


# If RoboComp was compiled with Python bindings you can use InnerModel in Python
# import librobocomp_qmat
# import librobocomp_osgviewer
# import librobocomp_innermodel

model = Model(
    wakeword_models=["shadow.onnx"],  # can also leave this argument empty to load all of the included pre-trained models
    inference_framework="onnx"
)

# Parameters
CHUNK = 1024  # Number of frames per buffer
FORMAT = pyaudio.paInt16  # Format of audio stream
CHANNELS = 1  # Number of audio channels
RATE = 16000  # Sampling rate
THRESHOLD = 5000  # Sound level threshold for activity detection
SILENCE_DURATION = 0.2  # Seconds of silence to stop recording

"""Record audio based on activity detection."""
p = pyaudio.PyAudio()

device_index = None
# List all available input devices
print("Available audio input devices:")
for i in range(p.get_device_count()):
    device_info = p.get_device_info_by_index(i)
    if device_info.get('maxInputChannels') > 0:  # Check if the device can handle input
        if "Jabra" in device_info['name']:
            device_index = i

if device_index is None:
    print("No Jabra found. Closing")
    exit(0)

stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                input_device_index=device_index,
                frames_per_buffer=CHUNK)

class SpecificWorker(GenericWorker):
    def __init__(self, proxy_map, startup_check=False):
        super(SpecificWorker, self).__init__(proxy_map)
        self.Period = 0
        if startup_check:
            self.startup_check()
        else:
            self.frames = np.array([], dtype=np.int16)  # Store frames as numpy array
            self.preroll_buffer = np.array([], dtype=np.int16)
            self.recording = False
            self.silence_start = None
            self.listening = False

            self.filename = "command.wav"
            print(whisper.available_models())
            self.model = whisper.load_model(name="turbo")

            # Precargar el archivo de ruido
            print(".................................................................")
            self.noise, self.noise_sr = self.load_noise("noise.wav")

            print("Listening for activity...")

            self.timer.timeout.connect(self.compute)
            self.timer.start(self.Period)

    def __del__(self):
        """Destructor"""

    def setParams(self, params):
        # try:
        #	self.innermodel = InnerModel(params["InnerModelPath"])
        # except:
        #	traceback.print_exc()
        #	print("Error reading config params")
        return True


    @QtCore.Slot()
    def compute(self):
        try:
            self.record_audio()
        except KeyboardInterrupt:
            # Close the stream
            stream.stop_stream()
            stream.close()
            p.terminate()
            console.print_exception(show_locals=True)
            exit(0)

    def record_audio(self):
        data = stream.read(CHUNK, exception_on_overflow=False)
        audio_data = np.frombuffer(data, dtype=np.int16)
        # Predict with the model (assuming it's a pre-trained model)
        prediction = model.predict(audio_data)
        if prediction["shadow"] > 0.5:
            print(prediction)
            self.listening = True

        if self.listening:
            if self.recording:
                self.frames = np.concatenate((self.frames, audio_data))
            else:
                self.preroll_buffer = np.concatenate((self.preroll_buffer, audio_data))  # Append to preroll array

            if not self.is_silent(data, THRESHOLD):
                if not self.recording:
                    print("Sound detected. Starting recording...")
                    self.recording = True
                    self.frames = np.array([])  # Reset frames for new recording
                self.silence_start = None
            else:
                if self.recording:
                    if self.silence_start is None:
                        self.silence_start = time.time()

                    elapsed_silence = time.time() - self.silence_start

                    if elapsed_silence > SILENCE_DURATION:
                        print("Silence detected. Stopping recording...")
                        total_data = np.concatenate((self.preroll_buffer, self.frames))
                        np_data = np.array(total_data, dtype=np.int16)


                        #
                        # Reset buffers for next recording
                        self.frames = np.array([])  # Clear frames for next recording
                        self.preroll_buffer = np.array([])
                        self.process_data(np_data)
                        self.listening = False
                        return

    # def record_audio(self):
    #     data = stream.read(CHUNK, exception_on_overflow=False)
    #     if self.recording:
    #         self.frames.append(data)
    #     else:
    #         self.preroll_buffer.append(data)
    #
    #     if not self.is_silent(data, THRESHOLD):
    #         if not self.recording:
    #             print("Sound detected. Starting recording...")
    #             self.recording = True
    #             self.frames = []  # Reset frames for new recording
    #         self.silence_start = None
    #     else:
    #         if self.recording:
    #             if self.silence_start is None:
    #                 self.silence_start = time.time()
    #
    #             elapsed_silence = time.time() - self.silence_start
    #
    #             if elapsed_silence > SILENCE_DURATION:
    #                 print("Silence detected. Stopping recording...")
    #                 preroll_list = list(self.preroll_buffer)
    #                 total_data = preroll_list + self.frames
    #                 self.process_data(total_data)
    #                 return

    def is_silent(self, data_chunk, threshold):
        """Check if the audio chunk is silent based on the threshold."""
        audio_data = np.frombuffer(data_chunk, dtype=np.int16)
        return np.max(np.abs(audio_data)) < threshold

    def process_data(self, data):
        # Save recorded audio to file
        filename = "command.wav"
        wf = wave.open(filename, 'wb')
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(p.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(data))
        wf.close()


        print(f"Audio saved to {filename}")

        # Aplicar reducción de ruido
        filtered_file = "filtered_command.wav"
        self.apply_noise_reduction(filename, filtered_file)

        # # Close the stream
        # stream.stop_stream()
        # stream.close()
        # p.terminate()

        result = self.model.transcribe(filtered_file, language="es")
        # result = self.model.transcribe("command.wav")
        print(result["text"])
        self.whisperstream_proxy.OnMessageTranscribed(result["text"].lower())

        self.recording = False
        self.silence_start = None

        print("Listening for activity...")

    def send_command(self, command):
        try:
            self.whisperstream_proxy.OnMessageTranscribed(command["text"])
        except Exception as e:
            console.print_exception(show_locals=True)
            print(f"Error sending command: {e}")

    def startup_check(self):
        QTimer.singleShot(200, QApplication.instance().quit)


    def apply_noise_reduction(self, input_file, output_file):
        """
        Aplica reducción de ruido usando el perfil precargado.
        """
        print("Aplicando reducción de ruido...")
        try:
            # Cargar archivo de entrada (señal con voz)
            voice, sr_voice = librosa.load(input_file, sr=None)

            # Verificar que las frecuencias de muestreo coincidan
            if sr_voice != self.noise_sr:
                raise ValueError("La frecuencia de muestreo del archivo de voz no coincide con la del ruido.")

            # Reducir ruido
            reduced_voice = nr.reduce_noise(y=voice, y_noise=self.noise, sr=sr_voice, prop_decrease=0.8)
            # Convertir el array de numpy a AudioSegment para normalizar
            reduced_voice_segment = AudioSegment(reduced_voice.tobytes(), frame_rate=sr_voice, sample_width=reduced_voice.dtype.itemsize,  channels=1)
            # Normalizar el volumen
            normalized_voice = reduced_voice_segment.apply_gain(-reduced_voice_segment.max_dBFS)
            # Guardar el archivo de audio resultante
            print(f"Guardando el resultado en {output_file}...")
            normalized_voice.export(output_file, format="wav")
            # Guardar resultado en archivo de salida
            sf.write(output_file, reduced_voice, sr_voice)
            print(f"Ruido reducido y guardado en: {output_file}")

        except Exception as e:
            print(f"Error aplicando reducción de ruido: {e}")
            raise

    def load_noise(self, noise_file):

        try:
            # Cargar archivo de ruido
            noise, original_sr = librosa.load(noise_file, sr=None)
            target_sr=RATE
            # Resamplear a la frecuencia deseada
            if original_sr != 16000:
                noise = librosa.resample(noise, orig_sr=original_sr, target_sr=16000)
                print(f"Perfil de ruido resampleado de {original_sr} Hz a {target_sr} Hz.")

            return noise, target_sr
        except Exception as e:
            print(f"Error cargando o resampleando el archivo de ruido: {e}")
            raise

    ######################
    # From the RoboCompWhisperStream you can call this methods:
    # self.whisperstream_proxy.OnMessageTranscribed(...)


