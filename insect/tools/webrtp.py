import asyncio
import cv2
import numpy as np
from aiortc import RTCPeerConnection, RTCSessionDescription, VideoStreamTrack
from av import VideoFrame
from flask import Flask, render_template, request, Response
import time
import Ice
Ice.loadSlice("Camera360RGB.ice")
import RoboCompCamera360RGB
import queue
import threading
import sys

data = Ice.InitializationData()
data.properties = Ice.createProperties()
data.properties.setProperty('Ice.MessageSizeMax', '20004800')  # Set to 10 MB

app = Flask(__name__)
pc = RTCPeerConnection()

class VideoImageTrack(VideoStreamTrack):
    def __init__(self):
        super().__init__()  # don't forget this!
        self.frame_queue = asyncio.Queue()

    async def recv(self):
        return await self.frame_queue.get()

video_track = VideoImageTrack()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/offer', methods=['POST'])
def offer():
    data = request.get_json()

    async def do_offer():
        await pc.setRemoteDescription(RTCSessionDescription(
            sdp=data['sdp'],
            type=data['type']
        ))

        for t in pc.getTransceivers():
            if t.kind == 'video' and t.receiver.track is not None:
                t.receiver.track.stop()
                pc.addTrack(video_track)

        answer = await pc.createAnswer()

        await pc.setLocalDescription(answer)

        return {'sdp': pc.localDescription.sdp, 'type': pc.localDescription.type}

    return asyncio.run(do_offer())

def run_ice_client():
    with Ice.initialize(sys.argv, data) as communicator:
        base = communicator.stringToProxy("camera360rgb:default -p 10097")
        camera = RoboCompCamera360RGB.Camera360RGBPrx.checkedCast(base)
        image = camera.getROI(-1, -1, -1, -1, -1, -1)
        frame = np.frombuffer(image.image, dtype=np.uint8).reshape((image.height, image.width, 3))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        video_frame = VideoFrame.from_ndarray(frame, format='rgb24')

        asyncio.run(video_track.frame_queue.put(video_frame))

if __name__ == '__main__':
    threading.Thread(target=run_ice_client).start()
    app.run(port=5000)

