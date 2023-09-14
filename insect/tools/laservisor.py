from flask import Flask, jsonify, send_from_directory, render_template, make_response
from flask_cors import CORS, cross_origin
import Ice
Ice.loadSlice("Lidar3D.ice")
import RoboCompLidar3D

app = Flask(__name__, static_url_path='')
CORS(app, origins='*', headers=['Content-Type'], methods=['GET'])


# Inicializar la conexión con el servidor Ice.
communicator = Ice.initialize()
base = communicator.stringToProxy("lidar3d:default -p 11988")
lidar = RoboCompLidar3D.Lidar3DPrx.checkedCast(base)
if not lidar:
    raise RuntimeError("Invalid proxy")

@app.route('/')
def serve_index():
    return render_template('indexlaser.html')

@app.route('/data', methods=['GET'])
#@cross_origin()  # Habilita CORS para la ruta /data
def serve_data():
    # Obtener los datos del LIDAR.
    data = lidar.getLidarData("helios", 0, 360, 1)

    # Transformar los datos a un formato JSON.

    json_data = [{'x': point.x/1000, 'y': point.y/1000, 'z': point.z/1000, 'intensity': 100} for point in data.points]
    response = make_response(jsonify(json_data))
    response.headers['Access-Control-Allow-Origin'] = '*'  # Permite todas las solicitudes desde cualquier origen

    return response
    
    #return jsonify(json_data)

if __name__ == '__main__':
    app.run(host='192.168.50.153', port=5000)

