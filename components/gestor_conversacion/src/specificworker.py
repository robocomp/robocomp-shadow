#!/usr/bin/python3
# -*- coding: utf-8 -*-
#
#    Copyright (C) 2021 by YOUR NAME HERE
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
from genericworker import *
import re
import os
sys.path.append('/opt/robocomp/lib')
console = Console(highlight=False)

# If RoboComp was compiled with Python bindings you can use InnerModel in Python
# import librobocomp_qmat
# import librobocomp_osgviewer
# import librobocomp_innermodel

from usb_4_mic_array.tuning import Tuning
import usb.core
import usb.util
import time
import speech_recognition as sr
import random
import json
from pixel_ring import pixel_ring

# Speech imports

max_queue = 100
charsToAvoid = ["'", '"', '{', '}', '[', '<', '>', '(', ')', '&', '$', '|', '#']
from google_speech import Speech
try:
	from Queue import Queue
except ImportError:
	from queue import Queue

# keywords and themes


temas = []
keyword = ["girafa", "jirafa", "hola"]
afirmaciones = ["okey", "vale", "perfe", "si", "guay","claro"]
negaciones = ["no", "que va", "para nada", "en absoluto"]
despedida = ["me voy", "adios"]
llamadas = ["aquí", "aqui", "ey", "ven", "vente", "gira"]

lines_path = os.path.join(os.path.dirname(__file__),'../lines')
for path in os.listdir(lines_path):
    if path != "start.json" and path != "end.json":
        y = path.replace('.json','')
        # print(y)
        temas.append(y)

# Microphone start

r = sr.Recognizer()
for i, microphone_name in enumerate(sr.Microphone.list_microphone_names()):
    print(microphone_name)
    if "ReSpeaker 4 Mic Array" in microphone_name:
        print("Micrófono seleccionado")
        m = sr.Microphone(device_index=i, sample_rate=16000)
        # m = sr.Microphone(device_index=i)

class Line:
    def __init__(self):
        self.next_possible_lines = []
        # self.past_line = ""
        self.line_name = ""
        self.phrase = ""
        self.emotion = ""
        self.was_talked = False
        self.is_binary = 0
        self.keywords = []
        self.to_start = 0
        self.needs_response = 0

    def show_past_action(self):
        print("- "+self.past_line)

    def show_next_possible_actions(self):
        for line in self.next_possible_lines:
            print("- "+line)

class SpecificWorker(GenericWorker):
    def __init__(self, proxy_map, startup_check=False):
        super(SpecificWorker, self).__init__(proxy_map)
        self.Period = 20
        if startup_check:
            self.startup_check()
        else:
            self.timer.timeout.connect(self.compute)
            self.timer.start(self.Period)
        self.user_name = ""
        self.lines_path = os.path.join(os.path.dirname(__file__),'../lines')

        with open(os.path.join(self.lines_path, "start.json"), "r") as f:
            start_line = json.loads(f.read())
            # print(start_line["next_possible_lines"])
            self.init_line = self.generate_line(start_line["line_name"], start_line["phrase"], start_line["next_possible_lines"], start_line["emotion"], start_line["binary"], start_line["keywords"], start_line["to_start"], start_line["needs_response"])
        with open(os.path.join(self.lines_path, "end.json"), "r") as f:
            end_line = json.loads(f.read())
            self.final_line = self.generate_line(end_line["line_name"], end_line["phrase"], end_line["next_possible_lines"], end_line["emotion"], end_line["binary"], end_line["keywords"], end_line["to_start"], end_line["needs_response"])
        # self.text_queue = Queue(max_queue)
        self.lines = []
        self.main_lines = []
        self.actual_line = {}
        self.elapsed_time = time.time()
        self.counter = 0
        self.exit = False
        self.isTalking = False
        self.isFollowing = False
        self.isBlocked = False
        self.process_queue = []
        self.isLost = False
        text = ",,, Hola. Puedo hablar."
        os.system("google_speech -l es "+ "'t " + text+ "'")

    def __del__(self):
        """Destructor"""

    def setParams(self, params):
        return True

    @QtCore.Slot()
    def compute(self):
        if not self.process_queue:
            pass
        else:
            print(self.process_queue)
            process = self.process_queue[0]
            if process[0] == 0:
                self.listenToHuman()
            elif process[0] == 1:
                self.follow_conversation(process[1])
            elif process[0] == 2:
                self.talk_conversation(process[1], process[2], process[3])
            elif process[0] == 3:
                self.not_follow_conversation(process[1])
            elif process[0] == 4:
                # if not self.isLost:
                self.lost_conversation(process[1])
                #     self.isLost = True
                # self.lost_process()
            elif process[0] == 5:
                self.sayHi_conversation(process[1])
            elif process[0] == 6:
                self.waiting_conversation(process[1])
            elif process[0] == 7:
                self.saySomething_conversation(process[1], process[2])
            if len(self.process_queue) > 0:
                self.process_queue.pop(0)

    def startup_check(self):
        QTimer.singleShot(200, QApplication.instance().quit)


    def person_position_converter(self, sound_pos):
        if 270 >= sound_pos >= 90:
            ang_deg = -(sound_pos - 90)
        elif sound_pos < 90:
            ang_deg = 90 - sound_pos
        else:
            ang_deg = sound_pos - 450
        return ang_deg

    def recorder(self):
        # data = input()
        # return data
        time.sleep(2)
        with m as source:
            try:
                r.adjust_for_ambient_noise(source)
                pixel_ring.set_brightness(10)
                pixel_ring.set_color(None, r=0, g=255, b=0)
                # print("Grabando")
                # self.emotionalmotor_proxy.listening(True)
                audio = r.listen(source, phrase_time_limit=3, timeout=2)
                # self.emotionalmotor_proxy.listening(False)
                pixel_ring.listen()
                record = r.recognize_google(audio, language="es-ES")
                return record
            except:
                pixel_ring.listen()
                return 0

    # For short answers (si, no)

    def recorder_binary(self):
        # data = input()
        # return data
        time.sleep(1)
        with m as source:
            r.adjust_for_ambient_noise(source)
            print("Grabando")
            pixel_ring.set_brightness(10)
            pixel_ring.set_color(None, r=0, g=255, b=0)
            # self.emotionalmotor_proxy.listening(True)
            audio = r.listen(source, phrase_time_limit=1.5)
            # self.emotionalmotor_proxy.listening(False)
            pixel_ring.listen()
            try:
                record = r.recognize_google(audio, language="es-ES")
                return record
            except:
                return 0

    def talker(self, text):
        # print(text)
        lang = "es"
        # speech = Speech(text, lang)
        # self.emotionalmotor_proxy.talking(True)
        # speech.save("act_audio.mp3")
        os.system("google_speech -l es "+ "' t " + text+ "'")
        # self.emotionalmotor_proxy.talking(False)
        return

    def generate_line(self, name, phrase, next_lines, emotion, is_binary, keywords, to_start, needs_response): 
        line = Line()
        line.next_possible_lines = next_lines
        # line.past_line = prev_line
        line.line_name = name
        line.phrase = phrase
        line.emotion = emotion
        line.is_binary = is_binary
        line.keywords = keywords
        line.to_start = to_start
        line.needs_response = needs_response
        return line

    def automatic_exit(self):
        self.talker("Vale. Hasta luego")
        self.exit = True  

    def inicio_lineas(self, tema):
        self.lines.clear()
        self.main_lines.clear()
        with open(os.path.join(self.lines_path, tema+".json"), "r") as f:
            line_json = json.loads(f.read())
            self.main_lines = line_json["main_lines"]

            # print(self.main_lines)

            for line in line_json["lines"]:
                line_generated = self.generate_line(line["line_name"], line["phrase"], line["next_possible_lines"], line["emotion"], line["binary"], line["keywords"], line["to_start"], line["needs_response"])        
                self.lines.append(line_generated)
        self.actual_line = self.lines[0] 
        self.talker(self.actual_line.phrase)
        while True:
            if(len(self.main_lines) == 0 and self.actual_line.to_start == 1):
                print("")
                state = self.agenteconversacional_proxy.componentState(0)
                print("fin conversacion")
                return

            if self.actual_line.needs_response == 0:
                pass
            else:
                if self.actual_line.is_binary == 1:
                    response = self.recorder_binary()
                else:
                    response = self.recorder()
                # print(response)

            if response in despedida:
                self.automatic_exit()
                return

            # print("")
            # print("To start: ", self.actual_line.to_start)
            # print("")
            # print("Len nets possserti: ", len(self.actual_line.next_possible_lines))
            # print("")
            # print("")   
            
            if self.actual_line.to_start == 1:
                # print("")
                # print("al inicio")
                # time.sleep(3)

                random_line = random.choice(self.main_lines)
                for line in self.lines:
                    if line.line_name == random_line:
                        self.actual_line = line 
                        self.main_lines.remove(line.line_name)
                        self.lines.remove(line)
                        break   
                self.talker(self.actual_line.phrase)  

            elif response == 0:
                self.talker("Perdona. No te he entendido.")  

            elif self.actual_line.to_start == 0 and len(self.actual_line.next_possible_lines) == 1:
                # print("")
                # print("solo una via")
                # time.sleep(3)
                for line in self.lines:
                    if line.line_name in self.actual_line.next_possible_lines:
                        self.actual_line = line
                        self.lines.remove(self.actual_line)
                        break
                time.sleep(0.5)
                self.talker(self.actual_line.phrase) 
            else:
                # print("")
                # print("varias lineas posibles")
                # time.sleep(3)
                max_i = 0
                for line in self.lines:
                    if line.line_name in self.actual_line.next_possible_lines:
                        split_response = response.split(" ")
                        i = 0                      
                        for x in split_response:
                            if(x in line.keywords):
                                # print(x)
                                i +=1
                        if i > max_i:
                            max_i = i
                            self.actual_line = line
                if i == 0:                    
                    self.talker(self.actual_line.phrase)
                else:
                    self.talker(self.actual_line.phrase)
                    self.lines.remove(self.actual_line) 
            # print("Nombre de la linea: "+self.actual_line.line_name)
            for i in self.lines:
                if i == self.actual_line:
                    pass
                    # print("")
                    # print("No se ha borrado")

##################################################################################################################
       
    def Conversation_listenToHuman(self):
        # for process in self.process_queue:
        #     if 0 in process:
        #         return
        self.process_queue.append([0])
 
    def Conversation_lost(self, name, role):
        for process in self.process_queue:
            if 4 in process:
                return
        self.process_queue.append([4, name, role])

    def Conversation_isBlocked(self, blocked):
        print("blocked: ",blocked)
        self.isBlocked = blocked

    def Conversation_isFollowing(self, following):
        print("following: ",following)
        self.isFollowing = following

    def Conversation_isTalking(self, talking):
        print("talking: ",talking)
        self.isTalking = talking

    def Conversation_talking(self, name, role, conversation):
        # elif conversation == "permiso":
        #     self.process_queue.append([1, name, role, conversation])
            # cadena = "Ey "+ name + ". ¿Puedes dejarme pasar, por favor?."
            # self.talker(cadena)
        if conversation == "hablar":
            self.process_queue.append([2, name, role, conversation])

    def Conversation_following(self, name, role):
        for process in self.process_queue:
            if 1 in process:
                return
        self.process_queue.append([1, name, role])
        
    def Conversation_saySomething(self, name, phrase):
        for process in self.process_queue:
            if 7 in process:
                return
        self.process_queue.append([7, name, phrase])

    def Conversation_stopFollowing(self, name, role):
        for process in self.process_queue:
            if 3 in process:
                return
        self.process_queue.append([3, name, role])

    def Conversation_sayHi(self, name, role):
        print("ENTRA")
        for process in self.process_queue:
            if 5 in process:
                return
        self.process_queue.append([5, name, role])
    
    def Conversation_waiting(self, name, role):
        print("ENTRA")
        for process in self.process_queue:
            if 6 in process:
                return
        self.process_queue.append([6, name, role])
    
    def listenToHuman(self):
        grabacion = self.recorder()
        if grabacion == 0:
            print("NO OYE NÁ")
            self.agenteconversacional_proxy.asynchronousIntentionReceiver(-99)
            return
        else:
            split_response = grabacion.split(" ")
            for x in split_response:
                if x == "sígueme":
                    print("sígueme")
                    self.agenteconversacional_proxy.asynchronousIntentionReceiver(0)
                    return
                elif x == "hola":
                    print("hola")
                    self.agenteconversacional_proxy.asynchronousIntentionReceiver(1)
                    return
                elif grabacion == "deja de seguirme":
                    print("deja de seguirme")
                    self.agenteconversacional_proxy.asynchronousIntentionReceiver(2) 
                    return
                elif grabacion == "espérame":
                    self.agenteconversacional_proxy.asynchronousIntentionReceiver(3) 
                    return
        print("NO OYE NÁ")
        self.agenteconversacional_proxy.asynchronousIntentionReceiver(-99)

    def waiting_conversation(self, name):
        cadena = "Vale. "+name+". Te espero"
        self.talker(cadena)

    def follow_conversation(self, name):
        cadena = "Te sigo."
        self.talker(cadena)
        
    def saySomething_conversation(self, name, phrase):
        self.talker(phrase)

    def not_follow_conversation(self, name):
        cadena = "Que vaya bien."
        self.talker(cadena)

    def permission_conversation(self, name):
        self.inicio_lineas("permission")

    def lost_conversation(self, name):
        cadena = "Espera " + name
        self.talker(cadena)

    def sayHi_conversation(self, name):
        cadena = "Hola. " + name + ".¿Qué necesitas?"
        self.talker(cadena)

    def lost_process(self):
        with m as source:
            r.adjust_for_ambient_noise(source)
            print("RECORDING")
            audio = r.listen(source, phrase_time_limit=3)
            try:
                record = r.recognize_google(audio, language="es-ES")
                split_response = record.split(" ")
                for word in split_response:
                    if word in llamadas:
                        print(record)
                        cadena = "voy"
                        self.talker(cadena)
                        self.agenteconversacional_proxy.asynchronousIntentionReceiver(3)
                        self.isLost = False
                        return
            except:
                print("no se ha entendido el audio")

    def talk_conversation(self, name, role, conversation):
        self.process_queue.pop(0)
        self.talker("Ey. "+name+". Que alegría verte.") 
        print(self.init_line.next_possible_lines)
        tema = random.choice(self.init_line.next_possible_lines)
        print(tema)
        self.inicio_lineas(tema)


    # ===================================================================
    # ===================================================================


    ######################
    # From the RoboCompEmotionalMotor you can call this methods:
    # self.emotionalmotor_proxy.expressAnger(...)
    # self.emotionalmotor_proxy.expressDisgust(...)
    # self.emotionalmotor_proxy.expressFear(...)
    # self.emotionalmotor_proxy.expressJoy(...)
    # self.emotionalmotor_proxy.expressSadness(...)
    # self.emotionalmotor_proxy.expressSurprise(...)
    # self.emotionalmotor_proxy.isanybodythere(...)
    # self.emotionalmotor_proxy.ispicture(...)
    # self.emotionalmotor_proxy.listening(...)
    # self.emotionalmotor_proxy.pupposition(...)
    # self.emotionalmotor_proxy.talking(...)

    ######################
    # From the RoboCompSoundRotation you can call this methods:
    # self.soundrotation_proxy.getKeyWord(...)
    # self.soundrotation_proxy.rotateAngle(...)

    ######################
    # From the RoboCompSpeech you can call this methods:
    # self.speech_proxy.isBusy(...)
    # self.speech_proxy.say(...)
    # self.speech_proxy.setPitch(...)
    # self.speech_proxy.setTempo(...)


