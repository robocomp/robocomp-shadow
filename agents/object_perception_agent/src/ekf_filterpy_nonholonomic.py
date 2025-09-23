import numpy as np
import matplotlib.pyplot as plt
from filterpy.kalman import ExtendedKalmanFilter
from filterpy.common import Q_discrete_white_noise
from numpy import dot
from scipy.linalg import block_diag

class NonHolonomicFilterPy:
    """
    Extended Kalman Filter for non-holonomic motion using FilterPy
    State: [x, y, theta, v] - position, heading, velocity
    Measurements: [x, y] - position only
    """
    
    def __init__(self, dt, process_noise_std, measurement_noise_std):
        """
        Initialize EKF using FilterPy
        
        Args:
            dt: Time step
            process_noise_std: Process noise standard deviation [pos, pos, angle, vel]
            measurement_noise_std: Measurement noise standard deviation [pos_x, pos_y]
        """
        self.dt = dt
        self.ekf = ExtendedKalmanFilter(dim_x=4, dim_z=2)
        
        # Initial state [x, y, theta, v]
        self.ekf.x = np.array([0., 0., 0., 1.0])
        
        # Initial covariance - large uncertainty
        self.ekf.P *= 1000
        
        # Process noise covariance matrix Q
        # We'll build this with individual variances for each state
        q_std = np.array(process_noise_std)
        self.ekf.Q = np.diag(q_std**2)
        
        # Measurement noise covariance matrix R
        r_std = np.array(measurement_noise_std)
        self.ekf.R = np.diag(r_std**2)
        
        # Store control input for use in jacobians
        self.control_input = np.array([0.0])  # Default: no turning
        
        # History for plotting
        self.history = {'x': [], 'P': [], 'residual': []}
    
    def motion_model(self, x, dt, u=None):
        """
        Non-holonomic motion model
        
        Args:
            x: state vector [x, y, theta, v]
            dt: time step
            u: control input [angular_velocity, acceleration] (optional)
        
        Returns:
            predicted state
        """
        if u is None:
            u = np.array([0.0])  # No control input
        
        # Extract state
        pos_x, pos_y, theta, v = x
        
        # Control inputs
        omega = u[0]  # angular velocity
        a = u[1] if len(u) > 1 else 0.0  # acceleration
        
        # Predict next state
        x_new = pos_x + v * np.cos(theta) * dt
        y_new = pos_y + v * np.sin(theta) * dt
        theta_new = theta + omega * dt
        v_new = v + a * dt
        
        return np.array([x_new, y_new, theta_new, v_new])
    
    def motion_jacobian(self, x, dt, u=None):
        """
        Jacobian of motion model with respect to state
        
        Args:
            x: state vector [x, y, theta, v]
            dt: time step
            u: control input (not used in jacobian)
        
        Returns:
            F: Jacobian matrix
        """
        pos_x, pos_y, theta, v = x
        
        F = np.array([
            [1, 0, -v * np.sin(theta) * dt, np.cos(theta) * dt],
            [0, 1,  v * np.cos(theta) * dt, np.sin(theta) * dt],
            [0, 0,  1,                      0],
            [0, 0,  0,                      1]
        ])
        
        return F
    
    def measurement_model(self, x):
        """
        Measurement model - we observe position only
        
        Args:
            x: state vector [x, y, theta, v]
        
        Returns:
            predicted measurement [x, y]
        """
        return np.array([x[0], x[1]])
    
    def measurement_jacobian(self, x):
        """
        Jacobian of measurement model with respect to state
        
        Args:
            x: state vector [x, y, theta, v]
        
        Returns:
            H: Measurement jacobian matrix
        """
        return np.array([[1, 0, 0, 0],
                        [0, 1, 0, 0]])
    
    def predict(self, control_input=None):
        """
        Prediction step
        
        Args:
            control_input: [angular_velocity] or [angular_velocity, acceleration]
        """
        if control_input is None:
            control_input = np.array([0.0])
        
        self.control_input = control_input
        
        # FilterPy predict step
        self.ekf.predict(
            lambda x, dt: self.motion_model(x, dt, control_input),
            lambda x, dt: self.motion_jacobian(x, dt, control_input),
            dt=self.dt
        )
        
        # Store history
        self.history['x'].append(self.ekf.x.copy())
        self.history['P'].append(self.ekf.P.copy())
    
    def update(self, measurement):
        """
        Update step with measurement
        
        Args:
            measurement: [x_measured, y_measured]
        """
        z = np.array(measurement)
        
        # FilterPy update step
        self.ekf.update(
            z,
            self.measurement_jacobian,
            self.measurement_model
        )
        
        # Store residual for analysis
        residual = z - self.measurement_model(self.ekf.x)
        self.history['residual'].append(residual)
    
    def predict_trajectory(self, n_steps, control_sequence=None):
        """
        Predict future trajectory
        
        Args:
            n_steps: Number of steps to predict
            control_sequence: List of control inputs for each step
        
        Returns:
            predictions: List of predicted states
            covariances: List of predicted covariances
        """
        # Save current state
        x_saved = self.ekf.x.copy()
        P_saved = self.ekf.P.copy()
        
        predictions = []
        covariances = []
        
        for i in range(n_steps):
            # Get control input for this step
            if control_sequence is not None and i < len(control_sequence):
                control = control_sequence[i]
            else:
                control = np.array([0.0])  # Default: no turning
            
            # Predict
            self.predict(control)
            
            # Store prediction
            predictions.append(self.ekf.x.copy())
            covariances.append(self.ekf.P.copy())
        
        # Restore original state
        self.ekf.x = x_saved
        self.ekf.P = P_saved
        
        return predictions, covariances
    
    @property
    def x(self):
        """Current state estimate"""
        return self.ekf.x
    
    @property
    def P(self):
        """Current covariance estimate"""
        return self.ekf.P
    
    def get_position_uncertainty(self):
        """
        Get current position uncertainty ellipse parameters
        
        Returns:
            center: [x, y] center position
            eigenvals: Eigenvalues of position covariance
            eigenvecs: Eigenvectors of position covariance
        """
        # Extract position covariance (2x2 submatrix)
        P_pos = self.ekf.P[:2, :2]
        
        # Eigenvalue decomposition
        eigenvals, eigenvecs = np.linalg.eigh(P_pos)
        
        return self.ekf.x[:2], eigenvals, eigenvecs


def simulate_nonholonomic_motion(dt, total_time, noise_std=0.1):
    """
    Simulate a person walking with non-holonomic constraints
    """
    times = np.arange(0, total_time, dt)
    n_steps = len(times)
    
    # True trajectory
    true_states = np.zeros((n_steps, 4))  # [x, y, theta, v]
    measurements = np.zeros((n_steps, 2))  # [x, y]
    control_inputs = []  # Store control inputs used
    
    # Initial state
    true_states[0] = [0, 0, 0, 1.0]  # Start at origin, facing east, 1 m/s
    
    for i in range(1, n_steps):
        # Generate control input (angular velocity)
        if i % 50 == 0:  # Turn every 5 seconds
            omega = np.random.choice([-0.5, 0.5])  # Turn left or right
        else:
            omega = 0.0
        
        # Occasionally change speed
        if i % 100 == 0:
            a = np.random.normal(0, 0.1)
        else:
            a = 0.0
        
        control_inputs.append(np.array([omega, a]))
        
        # Current state
        x, y, theta, v = true_states[i-1]
        
        # Update state using motion model
        x_new = x + v * np.cos(theta) * dt
        y_new = y + v * np.sin(theta) * dt
        theta_new = theta + omega * dt
        v_new = max(0.1, v + a * dt)  # Minimum velocity
        
        true_states[i] = [x_new, y_new, theta_new, v_new]
        
        # Add measurement noise
        measurements[i] = true_states[i, :2] + np.random.normal(0, noise_std, 2)
    
    return true_states, measurements, times, control_inputs


def plot_results_with_uncertainty(true_states, measurements, ekf_states, ekf_covariances, times):
    """
    Plot tracking results with uncertainty ellipses
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Trajectory plot with uncertainty ellipses
    ax = axes[0, 0]
    ax.plot(true_states[:, 0], true_states[:, 1], 'g-', label='True trajectory', linewidth=2)
    ax.scatter(measurements[:, 0], measurements[:, 1], c='r', s=10, alpha=0.3, label='Measurements')
    
    # EKF trajectory
    ekf_x = [s[0] for s in ekf_states]
    ekf_y = [s[1] for s in ekf_states]
    ax.plot(ekf_x, ekf_y, 'b-', label='EKF estimate', linewidth=2)
    
    # Plot uncertainty ellipses at regular intervals
    from matplotlib.patches import Ellipse
    import matplotlib.transforms as transforms
    
    for i in range(0, len(ekf_states), 50):  # Every 5 seconds
        # Get position covariance
        P_pos = ekf_covariances[i][:2, :2]
        
        # Eigenvalue decomposition
        eigenvals, eigenvecs = np.linalg.eigh(P_pos)
        
        # Calculate ellipse parameters (2-sigma bounds)
        angle = np.degrees(np.arctan2(eigenvecs[1, 0], eigenvecs[0, 0]))
        width = 2 * np.sqrt(eigenvals[0]) * 2  # 2-sigma
        height = 2 * np.sqrt(eigenvals[1]) * 2  # 2-sigma
        
        # Create ellipse
        ellipse = Ellipse((ekf_x[i], ekf_y[i]), width, height, angle=angle, 
                         alpha=0.3, facecolor='blue', edgecolor='blue')
        ax.add_patch(ellipse)
    
    ax.set_xlabel('X position (m)')
    ax.set_ylabel('Y position (m)')
    ax.set_title('Trajectory Tracking with Uncertainty Ellipses')
    ax.legend()
    ax.grid(True)
    ax.axis('equal')
    
    # Position errors
    ax = axes[0, 1]
    pos_errors = np.sqrt(np.sum((true_states[:, :2] - np.array([[s[0], s[1]] for s in ekf_states]))**2, axis=1))
    ax.plot(times, pos_errors, 'r-', linewidth=2, label='Position error')
    
    # Plot uncertainty bounds
    pos_uncertainty = [np.sqrt(np.trace(P[:2, :2])) for P in ekf_covariances]
    ax.fill_between(times, pos_errors - np.array(pos_uncertainty), 
                    pos_errors + np.array(pos_uncertainty), alpha=0.3, label='±1σ uncertainty')
    
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Position error (m)')
    ax.set_title('Position Error with Uncertainty')
    ax.legend()
    ax.grid(True)
    
    # Velocity tracking
    ax = axes[1, 0]
    ax.plot(times, true_states[:, 3], 'g-', label='True velocity', linewidth=2)
    ax.plot(times, [s[3] for s in ekf_states], 'b-', label='EKF estimate', linewidth=2)
    
    # Velocity uncertainty
    vel_uncertainty = [np.sqrt(P[3, 3]) for P in ekf_covariances]
    ekf_vel = [s[3] for s in ekf_states]
    ax.fill_between(times, np.array(ekf_vel) - np.array(vel_uncertainty),
                    np.array(ekf_vel) + np.array(vel_uncertainty), alpha=0.3, label='±1σ uncertainty')
    
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Velocity (m/s)')
    ax.set_title('Velocity Tracking with Uncertainty')
    ax.legend()
    ax.grid(True)
    
    # Heading tracking
    ax = axes[1, 1]
    ax.plot(times, true_states[:, 2], 'g-', label='True heading', linewidth=2)
    ax.plot(times, [s[2] for s in ekf_states], 'b-', label='EKF estimate', linewidth=2)
    
    # Heading uncertainty
    heading_uncertainty = [np.sqrt(P[2, 2]) for P in ekf_covariances]
    ekf_heading = [s[2] for s in ekf_states]
    ax.fill_between(times, np.array(ekf_heading) - np.array(heading_uncertainty),
                    np.array(ekf_heading) + np.array(heading_uncertainty), alpha=0.3, label='±1σ uncertainty')
    
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Heading (rad)')
    ax.set_title('Heading Tracking with Uncertainty')
    ax.legend()
    ax.grid(True)
    
    plt.tight_layout()
    plt.show()


def demonstrate_trajectory_prediction(ekf, true_states, times):
    """
    Demonstrate trajectory prediction capabilities
    """
    # Predict future trajectory
    n_future_steps = 30
    future_predictions, future_covariances = ekf.predict_trajectory(n_future_steps)
    
    # Create future time array
    future_times = np.arange(times[-1], times[-1] + n_future_steps * ekf.dt, ekf.dt)
    
    # Plot current trajectory and predictions
    plt.figure(figsize=(12, 8))
    
    # True trajectory
    plt.plot(true_states[:, 0], true_states[:, 1], 'g-', label='True trajectory', linewidth=2)
    
    # EKF estimated trajectory
    ekf_x = [s[0] for s in ekf.history['x']]
    ekf_y = [s[1] for s in ekf.history['x']]
    plt.plot(ekf_x, ekf_y, 'b-', label='EKF estimate', linewidth=2)
    
    # Predicted trajectory
    pred_x = [s[0] for s in future_predictions]
    pred_y = [s[1] for s in future_predictions]
    plt.plot(pred_x, pred_y, 'r--', label='Predicted trajectory', linewidth=2)
    
    # Prediction uncertainty ellipses
    from matplotlib.patches import Ellipse
    
    for i in range(0, len(future_predictions), 10):
        P_pos = future_covariances[i][:2, :2]
        eigenvals, eigenvecs = np.linalg.eigh(P_pos)
        
        angle = np.degrees(np.arctan2(eigenvecs[1, 0], eigenvecs[0, 0]))
        width = 2 * np.sqrt(eigenvals[0]) * 2  # 2-sigma
        height = 2 * np.sqrt(eigenvals[1]) * 2  # 2-sigma
        
        ellipse = Ellipse((pred_x[i], pred_y[i]), width, height, angle=angle,
                         alpha=0.2, facecolor='red', edgecolor='red')
        plt.gca().add_patch(ellipse)
    
    # Mark current position
    plt.plot(ekf.x[0], ekf.x[1], 'ko', markersize=10, label='Current position')
    
    plt.xlabel('X position (m)')
    plt.ylabel('Y position (m)')
    plt.title('Trajectory Prediction with Uncertainty')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    plt.show()


# Example usage
if __name__ == "__main__":
    # Check if filterpy is available
    try:
        from filterpy.kalman import ExtendedKalmanFilter
        print("FilterPy is available. Running demonstration...")
    except ImportError:
        print("FilterPy not installed. Install with: pip install filterpy")
        exit(1)
    
    # Simulation parameters
    dt = 0.1  # 10 Hz
    total_time = 30.0  # 30 seconds
    measurement_noise = 0.1  # 10 cm standard deviation
    
    # Generate simulated data
    print("Generating simulated trajectory...")
    true_states, measurements, times, control_inputs = simulate_nonholonomic_motion(
        dt, total_time, measurement_noise)
    
    # Initialize FilterPy EKF
    process_noise_std = [0.05, 0.05, 0.1, 0.1]  # [x, y, theta, v]
    measurement_noise_std = [0.1, 0.1]  # [x, y]
    
    ekf = NonHolonomicFilterPy(dt, process_noise_std, measurement_noise_std)
    
    # Set initial state
    ekf.ekf.x = np.array([0, 0, 0, 1.0])
    
    # Run EKF
    print("Running FilterPy EKF...")
    ekf_states = []
    ekf_covariances = []
    
    for i, (measurement, control) in enumerate(zip(measurements, control_inputs)):
        # Predict step
        ekf.predict(control)
        
        # Update with measurement
        ekf.update(measurement)
        
        # Store results
        ekf_states.append(ekf.x.copy())
        ekf_covariances.append(ekf.P.copy())
    
    # Plot results with uncertainty
    plot_results_with_uncertainty(true_states, measurements, ekf_states, ekf_covariances, times)
    
    # Demonstrate trajectory prediction
    demonstrate_trajectory_prediction(ekf, true_states, times)
    
    # Performance analysis
    final_pos_error = np.sqrt(np.sum((true_states[-1, :2] - ekf.x[:2])**2))
    print(f"\nFinal position error: {final_pos_error:.3f} m")
    
    mean_pos_error = np.mean(np.sqrt(np.sum((true_states[:, :2] - 
                                          np.array([[s[0], s[1]] for s in ekf_states]))**2, axis=1)))
    print(f"Mean position error: {mean_pos_error:.3f} m")
    
    print(f"Final state estimate: x={ekf.x[0]:.2f}, y={ekf.x[1]:.2f}, θ={ekf.x[2]:.2f}, v={ekf.x[3]:.2f}")
    print(f"Final uncertainty (position): {np.sqrt(ekf.P[0,0]):.3f} x {np.sqrt(ekf.P[1,1]):.3f} m")
    print(f"Final uncertainty (velocity): {np.sqrt(ekf.P[3,3]):.3f} m/s")
