import threading
from flask import Flask, Blueprint, jsonify, render_template, Response, request
from flask_cors import CORS, cross_origin
import Ice
Ice.loadSlice("Lidar3D.ice")
import RoboCompLidar3D
Ice.loadSlice("Camera360RGB.ice")
import RoboCompCamera360RGB

# Crear los blueprints
camera_blueprint = Blueprint('camera', __name__)
lidar_blueprint = Blueprint('lidar', __name__)
communicator_l = Ice.initialize()
base_l = communicator_l.stringToProxy("lidar3d:default -p 11988")
lidar = RoboCompLidar3D.Lidar3DPrx.checkedCast(base_l)

@camera_blueprint.route('/')
def index():
    width = request.args.get('width')
    height = request.args.get('height')
    return render_template('index-cam.html', width=width, height=height)

@camera_blueprint.route('/video_feed/<int:width>/<int:height>')
def video_feed(width, height):
    resolution = (width, height)
    return Response(generate(resolution), mimetype='multipart/x-mixed-replace; boundary=frame')

@lidar_blueprint.route('/')
def serve_index():
    return render_template('index_laser.html')

@lidar_blueprint.route('/data', methods=['GET'])
#@cross_origin()  # Habilita CORS para la ruta /data
def serve_data():
    # Obtener los datos del LIDAR.
    data = lidar.getLidarData("helios", 0, 360, 1)

    # Transformar los datos a un formato JSON.
    json_data = [
        {
            'x': point.x / 1000 if not math.isnan(point.x) else 10000,
            'y': point.y / 1000 if not math.isnan(point.y) else 10000,
            'z': point.z / 1000 if not math.isnan(point.z) else 10000,
            'intensity': 100
        }
        for point in data.points
    ]

    response = make_response(jsonify(json_data))
    response.headers['Access-Control-Allow-Origin'] = '*'  # Permite todas las solicitudes desde cualquier origen
    return response
    
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


# Inicializar la aplicación Flask y registrar los blueprints
app = Flask(__name__)
CORS(app, origins='*', headers=['Content-Type'], methods=['GET'])
app.register_blueprint(camera_blueprint, url_prefix='/camera')
app.register_blueprint(lidar_blueprint, url_prefix='/lidar')

if __name__ == "__main__":
    threading.Thread(target=run_ice_client).start()  # Esto podría necesitar ser movido o ajustado
    app.run(host='0.0.0.0', port=5000)


