import numpy as np


def alinear_habitacion_y_robot(esquinas, robot_pose, usar_centroide=True):
    """
    Recoloca las esquinas de una habitación Manhattan y la pose de un robot:
    - Alinea con los ejes X/Y
    - Centra en el origen (según centroide o bounding box)

    Parámetros:
        esquinas: lista de tuplas [(x1,y1), (x2,y2), ...]
        robot_pose: tupla (xr, yr, theta) posición (x,y) y orientación (rad)
        usar_centroide: bool, si True centra en el centroide geométrico,
                        si False centra en el centro del bounding box

    Devuelve:
        (puntos_final, robot_pose_final)
        donde puntos_final es np.array de las esquinas transformadas
        y robot_pose_final es np.array (x,y,theta) del robot transformado
    """
    puntos = np.array(esquinas)
    robot = np.array(robot_pose[:2])
    theta = robot_pose[2]

    # Calcular vectores de las aristas
    vectores = np.diff(np.vstack([puntos, puntos[0]]), axis=0)

    # Ángulos de los vectores
    angulos = np.arctan2(vectores[:, 1], vectores[:, 0])

    # Calcular ángulo dominante (mod 90°)
    angulos_mod = np.mod(angulos, np.pi / 2)
    angulo_rot = np.median(angulos_mod)

    # Matriz de rotación
    R = np.array([
        [np.cos(-angulo_rot), -np.sin(-angulo_rot)],
        [np.sin(-angulo_rot), np.cos(-angulo_rot)]
    ])

    # Rotar puntos y robot
    puntos_rot = puntos @ R.T
    robot_rot = robot @ R.T

    # Centrar en el origen
    if usar_centroide:
        centro = np.mean(puntos_rot, axis=0)
    else:
        centro = (np.min(puntos_rot, axis=0) + np.max(puntos_rot, axis=0)) / 2

    puntos_final = puntos_rot - centro
    robot_pos_final = robot_rot - centro

    # Orientación del robot: se rota en el mismo ángulo
    theta_final = theta - angulo_rot

    # Normalizar ángulo entre [-pi, pi]
    theta_final = (theta_final + np.pi) % (2 * np.pi) - np.pi

    return puntos_final, (robot_pos_final[0], robot_pos_final[1], theta_final)
