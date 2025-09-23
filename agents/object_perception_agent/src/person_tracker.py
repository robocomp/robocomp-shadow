import numpy as np


class EKFPersonTracker:
    def __init__(self):

        # Estado: [x, y, theta, vx, vy, omega]
        self.x = np.zeros((6, 1))

        # Covarianza del estado
        self.P = np.eye(6) * 0.1

        # Covarianza del proceso (ajustable)
        self.Q_base = np.diag([
            1e-4, 1e-4, 1e-3, 0.2, 0.2, 0.1  # px, py, theta, vx, vy, omega
        ])

        # Covarianza de la medición
        self.R = np.diag([
            0.0005, 0.0005, 0.01  # px, py en m, theta en rad
        ])

        # Matriz de observación
        self.H = np.array([
            [1, 0, 0, 0, 0, 0],  # px
            [0, 1, 0, 0, 0, 0],  # py
            [0, 0, 1, 0, 0, 0]  # theta
        ])

        self.last_timestamp = None
        self.last_dt = 0.1  # por defecto

    def initialize(self, z, timestamp):
        """Inicializa el filtro con la primera medición."""
        x, y, theta = z
        self.x = np.array([[x], [y], [self.normalize_angle(theta)], [0], [0], [0]])
        self.last_timestamp = timestamp

    def predict(self, dt):
        """Paso predictivo del filtro."""
        F = np.eye(6)
        F[0, 3] = dt  # x += vx * dt
        F[1, 4] = dt  # y += vy * dt
        F[2, 5] = dt  # theta += omega * dt

        self.x = F @ self.x
        self.x[2, 0] = self.normalize_angle(self.x[2, 0])

        Q = self.Q_base * dt
        self.P = F @ self.P @ F.T + Q

        self.last_dt = dt

    def update(self, z_mm, timestamp_ms):
        """Actualiza el filtro con una nueva medición z en mm y timestamp en ms."""
        z = np.array(z_mm, dtype=np.float64)
        z[:2] /= 1000.0  # mm → m
        timestamp = timestamp_ms / 1000.0  # ms → s

        if self.last_timestamp is None:
            self.initialize(z, timestamp)
            return

        dt = timestamp - self.last_timestamp
        if dt <= 0:
            dt = self.last_dt

        self.last_timestamp = timestamp

        self.predict(dt)

        z = z.reshape((3, 1))
        y = z - self.H @ self.x
        y[2, 0] = self.normalize_angle(y[2, 0])
        y[2, 0] = np.clip(y[2, 0], -0.5, 0.5)  # suavizado de saltos de orientación

        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)

        self.x = self.x + K @ y
        self.P = (np.eye(6) - K @ self.H) @ self.P

    def normalize_angle(self, angle):
        return (angle + np.pi) % (2 * np.pi) - np.pi

    def get_state(self):
        """Devuelve el estado actual: x, y, theta, vx, vy, omega"""
        return self.x.flatten()

    def get_pose(self):
        """Devuelve solo la pose filtrada: x, y, theta"""
        return self.x[0, 0], self.x[1, 0], self.x[2, 0]

    def get_velocity(self):
        """Devuelve las velocidades: vx, vy, omega"""
        return self.x[3, 0], self.x[4, 0], self.x[5, 0]
# from pykalman import KalmanFilter


# class PersonTracker:
#     def __init__(self, initial_position, initial_timestamp, initial_velocity=None):
#         """
#         Inicializa un tracker para una persona.
#
#         Args:
#             initial_position: np.array([x, y, z]) en metros.
#             initial_timestamp: Timestamp externo (float) en segundos.
#             initial_velocity: np.array([vx, vy, vz]) en m/s (opcional).
#         """
#         if initial_velocity is None:
#             initial_velocity = np.zeros(3)
#
#         # Estado del filtro de Kalman: [x, y, z, vx, vy, vz]
#         self.kf = KalmanFilter(
#             transition_matrices=self._get_transition_matrix(dt=1.0),  # Se actualizará en cada paso
#             observation_matrices=np.hstack([np.eye(3), np.zeros((3, 3))]),
#             initial_state_mean=np.hstack([initial_position, initial_velocity]),
#             initial_state_covariance=np.eye(6) * 10,
#             observation_covariance=np.eye(3) * 0.5,
#             transition_covariance=np.eye(6) * 0.1
#         )
#
#         self.last_state_mean = np.hstack([initial_position, initial_velocity])
#         self.last_state_cov = np.eye(6) * 10
#         self.last_update_time = initial_timestamp  # Usamos el timestamp proporcionado
#
#     def _get_transition_matrix(self, dt):
#         """Matriz de transición para un modelo de velocidad constante."""
#         return np.array([
#             [1, 0, 0, dt, 0, 0],
#             [0, 1, 0, 0, dt, 0],
#             [0, 0, 1, 0, 0, dt],
#             [0, 0, 0, 1, 0, 0],
#             [0, 0, 0, 0, 1, 0],
#             [0, 0, 0, 0, 0, 1]
#         ])
#
#     def update(self, new_position, new_timestamp):
#         """
#         Actualiza el tracker con una nueva posición y timestamp.
#
#         Args:
#             new_position: np.array([x, y, z]) en metros.
#             new_timestamp: Timestamp externo (float) en segundos.
#         """
#         dt = new_timestamp - self.last_update_time
#
#         # Actualizar matriz de transición con el dt real
#         self.kf.transition_matrices = self._get_transition_matrix(dt)
#
#         # Filtrar
#         self.last_state_mean, self.last_state_cov = self.kf.filter_update(
#             self.last_state_mean,
#             self.last_state_cov,
#             new_position
#         )
#         self.last_update_time = new_timestamp
#
#     def get_velocity(self):
#         """Devuelve la velocidad estimada como np.array([vx, vy, vz])."""
#         return self.last_state_mean[3:6]
#
#     def get_position(self):
#         """Devuelve la posición estimada como np.array([x, y, z])."""
#         return self.last_state_mean[:3]