from flask import Flask, render_template_string, request
import subprocess
import os
import signal
import psutil  # Necesitarás instalar esto con pip
from subprocess import Popen, PIPE

app = Flask(__name__)
processes = {}
commands = {
    'lidar3D': '~/robocomp/components/robocomp-robolab/components/hardware/laser/lidar3D/bin/Lidar3D ~/robocomp/components/robocomp-robolab/components/hardware/laser/lidar3D/etc/config_helios',
    'ricohOmni': '~/robocomp/components/robocomp-robolab/components/hardware/camera/ricoh_omni/bin/RicohOmni ~/robocomp/components/robocomp-robolab/components/hardware/camera/ricoh_omni/etc/config',
    'yolov8_360': 'python3 ~/robocomp/components/robocomp-shadow/insect/yolov8_360/src/yolov8_360.py ~/robocomp/components/robocomp-shadow/insect/yolov8_360/etc/config',
    'hashTracker': 'python3 ~/robocomp/components/robocomp-shadow/insect/hash_tracker/src/hash_tracker.py ~/robocomp/components/robocomp-shadow/insect/hash_tracker/etc/config_yolo',
    'gridPlanner': '~/robocomp/components/robocomp-shadow/insect/grid_planner/bin/grid_planner ~/robocomp/components/robocomp-shadow/insect/grid_planner/etc/config',
    'dwa': 'python3 ~/robocomp/components/robocomp-shadow/insect/dwa/src/dwa.py ~/robocomp/components/robocomp-shadow/insect/dwa/etc/config',
    'bumper': '~/robocomp/components/robocomp-shadow/insect/bumper/bin/bumper ~/robocomp/components/robocomp-shadow/insect/bumper/etc/config',
    'controller': 'python3 ~/robocomp/components/robocomp-shadow/insect/controller/src/controller.py ~/robocomp/components/robocomp-shadow/insect/controller/etc/config'
}

def kill_proc_tree(pid, including_parent=True):  
    parent = psutil.Process(pid)
    children = parent.children(recursive=True)
    for child in children:
        child.kill()
    if including_parent:
        parent.kill()
@app.route('/')
def home():
    template = """
    <html>
        <body>
            {% for command_id in commands.keys() %}
                <a href="{{ url_for('command_output', command_id=command_id) }}" target="_blank">{{ command_id }}</a><br/>
            {% endfor %}
        </body>
    </html>
    """
    return render_template_string(template, commands=commands)

@app.route('/<command_id>', methods=['GET', 'POST'])
def command_output(command_id):
    if request.method == 'POST':
        action = request.form.get('action')
        if action == 'start':
            command = commands.get(command_id)
            if command:
                process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True, shell=True)
                processes[command_id] = process
        elif action == 'stop':
            process = processes.get(command_id)
            print(process, type(process), command_id)
            
            if process:
                #os.kill(process.pid, signal.SIGKILL)  # envía SIGKILL al proceso del comando
                kill_proc_tree(process.pid)
                process.wait()
                del processes[command_id]
        return 'OK'

    output = ''
    process = processes.get(command_id)
    if process:
        try:
            output = process.communicate(timeout=1)[0]
        except subprocess.TimeoutExpired:
            pass

    template = """
    <html>
        <head>
            <meta http-equiv="refresh" content="5">
            <script src="https://ajax.googleapis.com/ajax/libs/jquery/3.6.0/jquery.min.js"></script>
            <script>
            $(function() {
                $('#start').click(function() {
                    $.post(window.location.href, {action: 'start'});
                });
                $('#stop').click(function() {
                    $.post(window.location.href, {action: 'stop'});
                });
            });
            </script>
        </head>
        <body>
            <h2>{{ command_id }}</h2>
            <button id="start">Start</button>
            <button id="stop">Stop</button>
            <pre>{{ output }}</pre>
        </body>
    </html>
    """
    return render_template_string(template, command_id=command_id, output=output)

if __name__ == "__main__":
    app.run(port=8080)
