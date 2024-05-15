import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Ellipse, Circle
import numpy as np

def parse_g2o_file(filename):
    """
    Reads a G2O file and creates Python dictionaries of vertex positions, landmarks,
    and edge information.

    Args:
        filename (str): 3D object file to be parsed.

    Returns:
        tuple: a triplet of dictionaries containing pose, landmark, and edge
        information for a given G2O file.

    """
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
    """
    Reads a file containing vertex IDs and matrix rows representing covariance
    matrices, parses them into Python objects, and returns a dictionary of covariances
    for each vertex.

    Args:
        filename (str): path to a file that contains the vertex IDs and their
            corresponding covariance matrices.

    Returns:
        dict: a dictionary of matrices, where each matrix represents the covariance
        between two vertices in the graph.

    """
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
    """
    Plots a graph representing robot poses and landmarks over time, with circles
    and orientation lines indicating the robot's position and orientation. Edges
    are also plotted using green lines. Optionally, covariance ellipses can be
    drawn for landmarks.

    Args:
        poses (ndarray of shape (N, 3), where N represents the number of poses.):
            3D robot poses as a list of lists, where each inner list contains the
            robot's x, y, and orientation (in radians) coordinates at a specific
            time step.
            
            		- `pose`: a 3-tuple containing the robot's position in a particular
            frame (e.g., world, body, or base) as (x, y, theta), where theta is
            the orientation of the robot in radians.
            		- `landmarks`: a dictionary containing the positions of the landmarks
            (features) in the environment, where each landmark is represented by
            a 2-tuple (x, y).
            		- `edges`: a dictionary containing the edges connected to each
            landmark, where each edge is represented by a tuple (vertex_id, x, y),
            where vertex_id is the ID of the adjacent landmark or robot.
            
            	The function explains how to plot each element of these inputs using
            matplotlib. The code comments describe the logic for drawing the poses
            as circles, orientation lines, and edges connected by line. Additionally,
            there are optional comments about drawing covariance ellipses for
            landmarks if appropriate and not original.
        landmarks (dict): 2D coordinates of landmarks that are used for plotting
            orientation lines and ellipses in the graph.
        edges (dict): 2D edges linked by line in the robot's workspace, which are
            plotted as green lines between paired vertex IDs.
        ax ("instance of mpl_toolkits.mplot3d.Axes3D".): 2D axes object where the
            plot will be drawn.
            
            		- `ax`: An instance of the Axes class in Matplotlib, used for plotting
            the graph.
            		- `title`: A string variable representing the title of the graph.
            		- `covariances`: An optional dictionary variable representing the
            covariances of the landmarks. If provided, it will be used to draw
            covariance ellipses for the landmarks.
            		- `original`: An optional boolean variable indicating whether the
            input poses are original or not. Used for drawing the orientation line.
            
            	Note: The function does not mention any properties of `ax`, as they
            are assumed to be properly initialized and available throughout the
            function. Therefore, no further explanation is provided.
        title (str): title that will be displayed at the top of the plot created
            by the function.
        covariances (dict): 2D covariance matrix of landmark positions, which can
            be optionally plotted as ellipses around each landmark for visualizing
            the spatial distribution of the landmarks.
        original (bool): initial value of the function, indicating whether or not
            to draw landmarks with ellipses representing covariances for each landmark.

    """
    for landmark in landmarks.values():
        ax.plot(landmark[0], landmark[1], 'ro')  # 'ro' for red circle

    # Plot poses with circles and orientation lines
    for pose in poses.values():
        it = 4
        y, x, theta = pose
        y = -y
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
        y, x, theta = pose
        y = -y
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
    """
    Reads and parses two G2O files, computes their covariances using a file
    containing the covariances, and plots both graphs with labels indicating which
    is the original and which is the optimized graph, using Matplotlib.

    """
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
