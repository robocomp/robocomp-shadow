import math
import numpy as np


def deg2rad(degrees):
    return degrees * math.pi / 180.0


def add_noise(value, std_dev):
    return value + np.random.normal(0, std_dev)


def main():
    """

    """
    num_steps = 50  # Number of steps in the circle
    angle_step = 360 / num_steps  # Angle step in degrees
    radius = 3.0  # Radius of the circle
    odometry_noise_std_dev = 0.1 # Standard deviation for odometry noise
    odometry_noise_angle_std_dev = 0.05  # Standard deviation for odometry noise
    measurement_noise_std_dev = 0.3  # Standard deviation for measurement noise

    # Define the landmarks (corners of the square)
    square_half_side = 6.0  # Half side of the square
    landmarks = [
        (square_half_side, square_half_side),
        (square_half_side, -square_half_side),
        (-square_half_side, square_half_side),
        (-square_half_side, -square_half_side)
    ]

    # Open the file to write the g2o content
    with open("circle.g2o", "w") as file:
        # Write the landmark vertices
        for i, (x, y) in enumerate(landmarks):
            file.write(f"VERTEX_XY {i} {x} {y}\n")

        last_x, last_y, last_theta = None, None, None
        # Write robot poses and observation edges with noise
        for i in range(num_steps):
            angle = deg2rad(i * angle_step)
            x = radius * math.cos(angle)
            y = radius * math.sin(angle)
            # theta = angle + math.pi / 2  # Robot faces along the tangent
            theta = angle - math.pi / 2 # Robot faces along the tangent

            noisy_x = add_noise(x, odometry_noise_std_dev)
            noisy_y = add_noise(y, odometry_noise_std_dev)
            noisy_theta = add_noise(theta, odometry_noise_angle_std_dev)

            # transform noisy_theta to be in the range [-pi, pi]
            noisy_theta = (noisy_theta + math.pi) % (2 * math.pi) - math.pi
            
            vertex_id = i + 4
            file.write(f"VERTEX_SE2 {vertex_id} {noisy_x} {noisy_y} {noisy_theta}\n")
            if i == 0:
                file.write(f"FIX {i + 4} \n")

            # Build transformation matrix from global system to robot system
            T = np.array([
                [math.cos(noisy_theta), -math.sin(noisy_theta), noisy_x],
                [math.sin(noisy_theta), math.cos(noisy_theta), noisy_y],
                [0, 0, 1]
            ])

            # Write odometry edges
            if vertex_id > 4:   # after first pose is fixed
                dx = noisy_x - last_x
                dy = noisy_y - last_y
                dtheta = noisy_theta - last_theta
                #add EDGE_SE2 to the file
                # file.write(f"EDGE_SE2 {vertex_id - 1} {vertex_id} {dx} {dy} {dtheta} {1/odometry_noise_std_dev} {0} {0} "
                #            f"{1/odometry_noise_std_dev} {0} {1/odometry_noise_angle_std_dev}\n")
                file.write(f"EDGE_SE2 {vertex_id - 1} {vertex_id} {dx} {dy} {dtheta} {0.1} {0} {0} "
                           f"{0.1} {0} {0.1}\n")

            last_x, last_y, last_theta = noisy_x, noisy_y, noisy_theta

            # Write noisy edges to each landmark
            for j, (lx, ly) in enumerate(landmarks):
                measured_dx = lx - x
                measured_dy = ly - y
                # Transform the measurements to the robot system using T i
                measured_dx, measured_dy, _ = np.dot(np.linalg.inv(T), [lx, ly, 1])

                noisy_dx = add_noise(measured_dx, measurement_noise_std_dev)
                noisy_dy = add_noise(measured_dy, measurement_noise_std_dev)
                # file.write(f"EDGE_SE2_XY {vertex_id} {j} {noisy_dx} {noisy_dy} {1/measurement_noise_std_dev} {0} {1.0/measurement_noise_std_dev}\n")
                file.write(f"EDGE_SE2_XY {vertex_id} {j} {noisy_dx} {noisy_dy} {1} {0} {1.0}\n")

    print("Generated g2o file with circular trajectory and noisy landmark observations.")


if __name__ == "__main__":
    main()
