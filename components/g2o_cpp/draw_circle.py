import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Ellipse, Circle
import numpy as np

def parse_g2o_file(filename):
    poses = {}
    landmarks = {}
    edges = {}
    with open(filename, 'r') as file:
        for line in file:
            parts = line.split()
            if parts[0] == 'VERTEX_SE2':
                vertex_id = int(parts[1])
                x = float(parts[2])
                y = float(parts[3])
                theta = float(parts[4])
                poses[vertex_id] = (x, y, theta)
            elif parts[0] == 'VERTEX_XY':
                vertex_id = int(parts[1])
                x = float(parts[2])
                y = float(parts[3])
                landmarks[vertex_id] = (x, y)
            elif parts[0] == 'EDGE_SE2_XY':
                vertex = int(parts[1])
                landmark = int(parts[2])
                x = float(parts[3])
                y = float(parts[4])
                edges[(vertex, landmark)] = (x, y)

    return poses, landmarks, edges


def read_covariances_from_file(filename):
    covariances = {}
    try:
        with open(filename, 'r') as file:
            lines = file.readlines()

        current_vertex = None
        for line in lines:
            if line.startswith('Vertex'):
                # Extract the vertex ID
                parts = line.split()
                current_vertex = int(parts[1])
            elif line.strip() and current_vertex is not None:
                # Read the matrix rows
                row1 = line.strip().split()
                row2 = line.strip().split()
                print(row1, row2)
                # Convert strings to floats and create the matrix
                matrix = [
                    [float(row1[0]), float(row1[1])],
                    [float(row2[1]), float(row2[0])]
                ]
                covariances[current_vertex] = matrix
                current_vertex = None  # Reset the vertex to ensure matrix pairs are processed correctly

    except FileNotFoundError:
        print(f"File not found: {filename}")
    except Exception as e:
        print(f"An error occurred: {e}")

    return covariances

def plot_graph(poses, landmarks, edges, ax, title, covariances=None, original=True):
    # Plot landmarks
    for landmark in landmarks.values():
        ax.plot(landmark[0], landmark[1], 'ro')  # 'ro' for red circle

    # Plot poses with circles and orientation lines
    for pose in poses.values():
        it = 4
        x, y, theta = pose
        # Create a circle at the robot's position
        circle = Circle((x, y), radius=20, color='blue', fill=True)
        ax.add_patch(circle)
        # Draw the orientation line
        # line_length = 60
        # end_x = x + line_length * np.cos(theta)
        # end_y = y + line_length * np.sin(theta)
        # plot una cada cinco posiciones como una flecha roja
        # if int(x) % 5 == 0:
        # ax.arrow(x, y, end_x - x, end_y - y, head_width=5, head_length=10, fc='r', ec='r')
        # ax.plot([x, end_x], [y, end_y], 'k-')  # 'k-' for black line
        # for i in range(4):
        #     edge_data = edges.get((it, i))
        #     if edge_data:
        #         print(edge_data)
        #         x, y = edge_data
        #         ax.plot([poses[it][0], x], [poses[it][1], y], 'o')
        it += 1

    for pose in poses.values():
        x, y, theta = pose
        # Create a circle at the robot's position
        # circle = Circle((x, y), radius=20, color='blue', fill=True)
        # ax.add_patch(circle)
        # Draw the orientation line
        line_length = 60
        end_x = x + line_length * np.cos(theta)
        end_y = y + line_length * np.sin(theta)
        # plot una cada cinco posiciones como una flecha roja
        # if int(x) % 5 == 0:
        ax.arrow(x, y, end_x - x, end_y - y, head_width=5, head_length=10, fc='r', ec='r')

    # Plot edges linked by line

    # for i, edges in edges.items():
    #     ax.plot([poses[vertex_id][0], x], [poses[vertex_id][1], y], 'g-')  # 'g-' for green line


    # Optionally, draw covariance ellipses for landmarks
    if covariances and not original:
        for vertex_id, cov in covariances.items():
            if vertex_id in landmarks:
                x, y = landmarks[vertex_id]
                draw_covariance_ellipse(x, y, cov, ax)

    ax.set_xlabel('X position')
    ax.set_ylabel('Y position')
    ax.set_title(title)
    ax.axis('equal')

def triangle_points(x, y, theta, scale=0.2):
    """ Generate vertices of the triangle representing the robot's pose and orientation. """
    return [
        (x + scale * np.cos(theta), y + scale * np.sin(theta)),  # tip of the triangle
        (x + scale * np.cos(theta + 2*np.pi/3), y + scale * np.sin(theta + 2*np.pi/3)),  # left base
        (x + scale * np.cos(theta - 2*np.pi/3), y + scale * np.sin(theta - 2*np.pi/3))   # right base
    ]

def draw_covariance_ellipse(x, y, cov, ax):
    """ Draw an ellipse based on the covariance matrix. """
    lambda_, v = np.linalg.eig(cov)
    lambda_ = np.sqrt(lambda_)
    ellipse = Ellipse(xy=(x, y), width=lambda_[0]*2, height=lambda_[1]*2,
                      angle=np.degrees(np.arccos(v[0, 0])), edgecolor='red', facecolor='none')
    ax.add_patch(ellipse)
# def plot_graph(poses, landmarks, ax, title, covariances, original):
#     # Plot poses
#     for pose in poses.values():
#         ax.plot(pose[0], pose[1], 'bo')  # 'bo' for blue circle
#     # Plot landmarks
#     # add now covariances in an array of 2x2 matrices
#     # Manually defined covariance matrices for each landmark
#
#     for i, (x, y) in landmarks.items():
#         ax.plot(x, y, 'ro')  # 'ro' for red circle
#         if not original and covariances and i in covariances:
#             # Add covariance ellipse
#             cov = covariances[i]
#             lambda_, v = np.linalg.eig(cov)
#             lambda_ = np.sqrt(lambda_)
#             ell = Ellipse(xy=(x, y),
#                           width=lambda_[0] * 2, height=lambda_[1] * 2,
#                           angle=np.rad2deg(np.arccos(v[0, 0])),
#                           edgecolor='red', facecolor='none')
#             ax.add_patch(ell)
#
#     # Optionally, connect pose points
#     pose_sequence = list(poses.values())
#     xs, ys = zip(*pose_sequence)
#     ax.plot(xs, ys, 'b-', alpha=0.5)  # Connect poses with a blue line
#
#     # Set labels and title
#     ax.set_xlabel('X position')
#     ax.set_ylabel('Y position')
#     ax.set_title(title)
#     ax.axis('equal')  # Equal scaling of the x and y axes

def main():
    # Filenames for the original and optimized graphs
    original_filename = 'trajectory.g2o'  # Update this to your original .g2o file path
    optimized_filename = 'optimized_trajectory.g2o'  # Update this to your optimized .g2o file path

    # Parse both files
    original_poses, original_landmarks, original_edges = parse_g2o_file(original_filename)
    optimized_poses, optimized_landmarks, optimized_edges = parse_g2o_file(optimized_filename)

    # Setup matplotlib figures and axes
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    covariances = read_covariances_from_file('covariances.txt')

    # Plot graphs
    plot_graph(original_poses, original_landmarks, original_edges, ax1, 'Original Graph', covariances, original=True)
    plot_graph(optimized_poses, optimized_landmarks, optimized_edges, ax2, 'Optimized Graph', covariances, original=False)

    # Show the plot
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
