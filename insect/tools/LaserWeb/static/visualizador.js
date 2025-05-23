// Importar Three.js
import * as THREE from './three.module.js';
// Importar OrbitControls
import { OrbitControls } from './OrbitControls.js';

// Crear escena, cámara, renderizador...
var scene = new THREE.Scene();
var camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
var renderer = new THREE.WebGLRenderer();

renderer.setSize(window.innerWidth, window.innerHeight);
document.body.appendChild(renderer.domElement);

// Crear controles orbitales
var controls = new OrbitControls(camera, renderer.domElement);

camera.position.z = 5;

// Crear un material y geometría para los puntos.
var material = new THREE.PointsMaterial({ vertexColors: true, size: 0.05 });
var geometry = new THREE.BufferGeometry();

// Añadir los puntos a la escena.
var points = new THREE.Points(geometry, material);
scene.add(points);

// Crear una función para actualizar los puntos.
function updatePoints(data) {
    // Crear un nuevo Float32Array para las posiciones de los puntos.
    var positions = new Float32Array(data.length * 3);
    var colors = new Float32Array(data.length * 3);

    // Llenar el array de posiciones con los datos de los puntos.
    for (var i = 0; i < data.length; i++) {
        positions[i * 3] = data[i].x;
        positions[i * 3 + 1] = data[i].y;
        positions[i * 3 + 2] = data[i].z;
        var intensity = data[i].intensity / 255.0;  // Normalizar intensidad al rango 0-1
        colors[i * 3] = intensity;
        colors[i * 3 + 1] = intensity;
        colors[i * 3 + 2] = intensity;
    }

    // Actualizar la geometría de los puntos con las nuevas posiciones y colores.
    points.geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    points.geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));
    points.geometry.computeBoundingSphere();
}

// Crear un intervalo para solicitar y actualizar los datos cada segundo.
setInterval(() => {
    fetch('http://192.168.50.153:5000/data')
        .then(response => response.json())
        .then(data => {
            updatePoints(data);
        });
}, 1000);

function animate() {
    requestAnimationFrame(animate);
    controls.update();  // Actualizar los controles en cada frame
    renderer.render(scene, camera);
}

animate();

