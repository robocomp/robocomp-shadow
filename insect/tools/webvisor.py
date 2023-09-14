from flask import Flask, request, render_template_string, Response
import subprocess
import threading
import queue
import os
import signal
import psutil
from collections import defaultdict

app = Flask(__name__)

commands = {
    'Lidar3D': '~/robocomp/components/robocomp-robolab/components/hardware/laser/lidar3D/bin/Lidar3D ~/robocomp/components/robocomp-robolab/components/hardware/laser/lidar3D/etc/config_helios_jetson',
    'RicohOmni': '~/robocomp/components/robocomp-robolab/components/hardware/camera/ricoh_omni/bin/RicohOmni ~/robocomp/components/robocomp-robolab/components/hardware/camera/ricoh_omni/etc/config',
    'yolov8_360': 'python3 ~/robocomp/components/robocomp-shadow/insect/yolov8_360/src/yolov8_360.py ~/robocomp/components/robocomp-shadow/insect/yolov8_360/etc/config',
    'hashTracker': 'python3 ~/robocomp/components/robocomp-shadow/insect/hash_tracker/src/hash_tracker.py ~/robocomp/components/robocomp-shadow/insect/hash_tracker/etc/config_yolo',
    'gridPlanner': '~/robocomp/components/robocomp-shadow/insect/grid_planner/bin/grid_planner ~/robocomp/components/robocomp-shadow/insect/grid_planner/etc/config',
    'dwa': 'python3 ~/robocomp/components/robocomp-shadow/insect/dwa/src/dwa.py ~/robocomp/components/robocomp-shadow/insect/dwa/etc/config',
    'bumper': '~/robocomp/components/robocomp-shadow/insect/bumper/bin/bumper ~/robocomp/components/robocomp-shadow/insect/bumper/etc/config',
    'controller': 'python3 ~/robocomp/components/robocomp-shadow/insect/controller/src/controller.py ~/robocomp/components/robocomp-shadow/insect/controller/etc/config'
}

processes = {}
queues = defaultdict(queue.Queue)

def kill_proc_tree(pid, including_parent=True):  
    parent = psutil.Process(pid)
    children = parent.children(recursive=True)
    for child in children:
        child.kill()
    if including_parent:
        parent.kill()

@app.route('/')
def home():
    return render_template_string(''.join(['<a href="{{ url_for(\'command_output\', command_id=\'%s\') }}" target="_blank">%s</a><br/>' % (command_id, command_id) for command_id in commands.keys()]))


@app.route('/<command_id>')
def command_output(command_id):
    return render_template_string("""
    <h1>{{ command_id }}</h1>
    <button onclick="start()">Start</button>
    <button onclick="stop()">Stop</button>
    <div id="output"></div>
    <script src="https://ajax.googleapis.com/ajax/libs/jquery/3.5.1/jquery.min.js"></script>
    <script>
    function start() {
        $.post("/start_stop/" + "{{ command_id }}", { action: "start" });
    }
    function stop() {
        $.post("/start_stop/" + "{{ command_id }}", { action: "stop" });
    }
    const eventSource = new EventSource("/stream/" + "{{ command_id }}");
    eventSource.onmessage = function(event) {
        document.getElementById("output").innerHTML += event.data + "<br/>";
    };
    </script>
    """, command_id=command_id)

@app.route('/start_stop/<command_id>', methods=['POST'])
def start_stop(command_id):
    action = request.form.get('action')
    if action == 'start':
        if command_id not in processes:
            command = commands[command_id]
            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, shell=True, text=True)
            def stream_output(process, queue):
                for line in iter(process.stdout.readline, ''):
                    queue.put(line)
                process.stdout.close()
            thread = threading.Thread(target=stream_output, args=(process, queues[command_id]))
            thread.start()
            processes[command_id] = process
    elif action == 'stop':
        process = processes.get(command_id)
        if process:
            kill_proc_tree(process.pid)
            del processes[command_id]
    return ''

@app.route('/stream/<command_id>')
def stream(command_id):
    def event_stream():
        while True:
            yield 'data: {}\n\n'.format(queues[command_id].get())
    return Response(event_stream(), mimetype="text/event-stream")

if __name__ == "__main__":
    app.run(port=8080, threaded=True)

