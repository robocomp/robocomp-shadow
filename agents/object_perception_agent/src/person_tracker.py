import numpy as np
from numba import jit


# We extract the heavy math outside the class to allow Numba to compile it.
# nopython=True ensures it compiles to machine code (no Python interpreter).
# cache=True saves the compiled code to disk so restart is fast.
@jit(nopython=True, cache=True, fastmath=True)
def ekf_predict_update(x, P, z, dt, Q_base, R):
    # --- PREDICT STEP ---
    # Construct F matrix manually to avoid eye() overhead
    # F is 6x6 Identity with dt in specific spots
    # State: [x, y, theta, vx, vy, omega]

    # x = F @ x
    x[0, 0] += x[3, 0] * dt
    x[1, 0] += x[4, 0] * dt
    x[2, 0] += x[5, 0] * dt

    # Normalize theta (Predict)
    x[2, 0] = (x[2, 0] + np.pi) % (2 * np.pi) - np.pi

    # P = F @ P @ F.T + Q
    # We construct the Jacobian F explicitly for P update
    # or better yet, since F is sparse, we do it manually or via multiplication
    # For simplicity and Numba speed, standard multiplication is fine here
    F = np.eye(6)
    F[0, 3] = dt
    F[1, 4] = dt
    F[2, 5] = dt

    Q = Q_base * dt
    P = F @ P @ F.T + Q

    # --- UPDATE STEP ---
    # z is [x, y, theta]
    # H is identity for first 3 rows. We optimize H @ x by just taking x[:3]

    # y = z - H @ x
    y_residual = z - x[:3]

    # Normalize theta (Innovation)
    y_residual[2, 0] = (y_residual[2, 0] + np.pi) % (2 * np.pi) - np.pi

    # Clip logic from original code
    if y_residual[2, 0] > 0.5:
        y_residual[2, 0] = 0.5
    elif y_residual[2, 0] < -0.5:
        y_residual[2, 0] = -0.5

    # S = H @ P @ H.T + R
    # H is a projection of the top left 3x3. We can just slice P.
    S = P[:3, :3] + R

    # K = P @ H.T @ inv(S)
    # P @ H.T is equivalent to taking the first 3 columns of P
    P_Ht = P[:, :3]

    # Use solve instead of inv for speed and stability
    # K = P_Ht @ inv(S) -> Solve S * K.T = P_Ht.T -> K = (Solve S, P_Ht.T).T
    # However, for 3x3, straightforward solve is:
    K = P_Ht @ np.linalg.inv(S)

    # x = x + K @ y
    x = x + K @ y_residual

    # P = (I - K @ H) @ P
    # Optimized: P = P - K @ (H @ P)
    # H @ P is just P[:3, :]
    P = P - K @ P[:3, :]

    return x, P


class EKFPersonTracker:
    def __init__(self):
        # State: [x, y, theta, vx, vy, omega]
        self.x = np.zeros((6, 1))

        # Covariance
        self.P = np.eye(6) * 0.1

        # Process Noise
        self.Q_base = np.diag(np.array([
            1e-4, 1e-4, 1e-3, 0.2, 0.2, 0.1
        ]))

        # Measurement Noise
        self.R = np.diag(np.array([
            0.0005, 0.0005, 0.01
        ]))

        self.last_timestamp = None
        self.last_dt = 0.1

    def initialize(self, z, timestamp):
        x_val, y_val, theta = z
        self.x = np.array([[x_val], [y_val], [self.normalize_angle(theta)], [0], [0], [0]])
        self.last_timestamp = timestamp

    def update(self, z_mm, timestamp_ms):
        """Actualiza el filtro con una nueva medición z en mm y timestamp en ms."""
        # Convert to numpy array float64 immediately
        z_input = np.array(z_mm, dtype=np.float64)

        # Unit conversion
        z_x = z_input[0] / 1000.0
        z_y = z_input[1] / 1000.0
        z_theta = z_input[2]
        timestamp = timestamp_ms / 1000.0

        if self.last_timestamp is None:
            self.initialize([z_x, z_y, z_theta], timestamp)
            return

        dt = timestamp - self.last_timestamp
        if dt <= 0:
            dt = self.last_dt

        self.last_timestamp = timestamp
        self.last_dt = dt

        # Prepare measurement vector (3x1)
        z = np.array([[z_x], [z_y], [z_theta]])

        # CALL THE JIT COMPILED FUNCTION
        # We pass numpy arrays. The first time this runs, it takes ~200ms to compile.
        # Every subsequent run will be microseconds.
        self.x, self.P = ekf_predict_update(
            self.x,
            self.P,
            z,
            dt,
            self.Q_base,
            self.R
        )

    def normalize_angle(self, angle):
        return (angle + np.pi) % (2 * np.pi) - np.pi

    def get_state(self):
        return self.x.flatten()

    def get_pose(self):
        return self.x[0, 0], self.x[1, 0], self.x[2, 0]

    def get_velocity(self):
        return self.x[3, 0], self.x[4, 0], self.x[5, 0]