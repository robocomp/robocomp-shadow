import math
import numpy as np


def deg2rad(degrees):
    return degrees * math.pi / 180.0


def add_noise(value, std_dev):
    return value + np.random.normal(0, std_dev)


def main():
    num_steps = 50  # Number of steps in the circle
    angle_step = 360/num_steps  # Angle step in degrees
    radius = 3.0  # Radius of the circle
    odometry_noise_std_dev = 0.1  # Standard deviation for odometry noise
    odometry_noise_ang_std_dev = 0.01 # Standard deviation for odometry noise in angle (rads)
    measurement_noise_std_dev = 0.05  # Standard deviation for measurement noise

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

        # Write robot poses and observation edges with noise
        for i in range(num_steps):
            angle = deg2rad(i*angle_step)
            x = radius * math.cos(angle)
            y = radius * math.sin(angle)
            theta = angle + math.pi / 2  # Robot faces along the tangent

            # Add noise to odometry (pose)
            noisy_x = add_noise(x, odometry_noise_std_dev)
            noisy_y = add_noise(y, odometry_noise_std_dev)
            noisy_theta = add_noise(theta, odometry_noise_ang_std_dev)

            # Write the robot's noisy pose vertex
            file.write(f"VERTEX_SE2 {i + 4} {noisy_x} {noisy_y} {noisy_theta}\n")
            if i == 0:
                file.write(f"FIX {i + 4} \n")

            # Write noisy edges to previous pose
            file.write(f"EDGE_SE2 {i + 3} {i + 4} {x} {y} {theta} {1.0/odometry_noise_std_dev} {0} {0} {1.0/odometry_noise_std_dev} {0} {1.0/odometry_noise_ang_std_dev}\n")

            # Write noisy edges to each landmark
            for j, (lx, ly) in enumerate(landmarks):
                measured_dx = lx - x
                measured_dy = ly - y
                noisy_dx = add_noise(measured_dx, measurement_noise_std_dev)
                noisy_dy = add_noise(measured_dy, measurement_noise_std_dev)
                file.write(f"EDGE_SE2_XY {i + 4} {j} {noisy_dx} {noisy_dy} {1.0/measurement_noise_std_dev} {0} {1.0/measurement_noise_std_dev}\n")

    print("Generated g2o file with circular trajectory and noisy landmark observations.")


if __name__ == "__main__":
    main()
