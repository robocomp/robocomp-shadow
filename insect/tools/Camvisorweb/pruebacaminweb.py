import time
import Ice
import cv2
import IceStorm
from rich.console import Console, Text
console = Console()
Ice.loadSlice("Camera360RGB.ice")
import RoboCompCamera360RGB
import numpy as np
import sys
import queue


data = Ice.InitializationData()
data.properties = Ice.createProperties()
data.properties.setProperty('Ice.MessageSizeMax', '20004800')  # Set to 10 MB

import queue
from flask import Flask, Response, render_template, request
import threading

frame_queue = queue.Queue()
frame_lock = threading.Lock()

app = Flask(__name__)


MAX_QUEUE_SIZE = 2  # ajustar según sea necesario

@app.route('/')
def index():
    width = request.args.get('width')
    height = request.args.get('height')
    return render_template('index.html', width=width, height=height)



@app.route('/video_feed/<int:width>/<int:height>')
def video_feed(width, height):
    resolution = (width, height)
    return Response(generate(resolution), mimetype='multipart/x-mixed-replace; boundary=frame')


def generate(resolution):
    while True:
        try:
            with frame_lock:
                frame = frame_queue.get_nowait()

            # Redimensionar el marco a la resolución deseada
            frame = cv2.resize(frame, resolution)

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            _, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 100])

            frame = buffer.tobytes()

            yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        except queue.Empty:
            print("Queue is empty, skipping...")
            continue


MAX_QUEUE_SIZE = 2  # ajustar según sea necesario

def run_ice_client():
    with Ice.initialize(sys.argv, data) as communicator:
        base = communicator.stringToProxy("camera360rgb:default -p 10097")
        camera = RoboCompCamera360RGB.Camera360RGBPrx.checkedCast(base)
        if not camera:
            raise RuntimeError("Invalid proxy")

        while True:
            image = camera.getROI(-1, -1, -1, -1, 640, 640)

            with frame_lock:
                if frame_queue.qsize() < MAX_QUEUE_SIZE:
                    frame_queue.put(np.frombuffer(image.image, dtype=np.uint8).reshape((image.height, image.width, 3)))
                else:
                    # descarta la imagen más antigua si la cola está demasiado llena
                    frame_queue.get()
                    frame_queue.put(np.frombuffer(image.image, dtype=np.uint8).reshape((image.height, image.width, 3)))
            
            
            
            
            
                

if __name__ == '__main__':
    threading.Thread(target=run_ice_client).start()
    app.run(port=5000)

