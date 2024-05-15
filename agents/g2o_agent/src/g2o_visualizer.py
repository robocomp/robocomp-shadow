import matplotlib.pyplot as plt
import g2o
import time
import numpy as np
from matplotlib.patches import Polygon, Ellipse, Circle
import math

class G2OVisualizer:
    def __init__(self, title):
        """
        Sets up a graphical interactive window with Matplotlib and configures its
        title, x-axis limits, and y-axis limits.

        Args:
            title (str): title of the subplot, which is then displayed on the plot
                using the `set_title()` method.

        """
        self.title = title
        
        # Configurar la ventana gráfica interactiva de Matplotlib
        plt.ion()
        self.fig, self.ax = plt.subplots()

        self.ax.set_title(self.title)
        self.ax.set_aspect('equal')
        self.ax.legend()
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

        # Fijar límites de los ejes
        # self.ax.set_xlim([-3000, 3000])  # Por ejemplo, límites para el eje x de 0 a 10
        # self.ax.set_ylim([-3000, 3000])  # Por ejemplo, límites para el eje y de 0 a 10

    def edges_coord(self, edges, dim):
        """
        Generates coordinates for the edges in a graph by returning the x-y
        coordinates of each vertex in the edge.

        Args:
            edges (list): 2D or 3D edges of a shape, from which the coordinate
                values of each edge are estimated and returned.
            dim (int): dimension of the coordinate system used to estimate the
                coordinates of the vertices in each edge.

        """
        for e in edges:
            yield e.vertices()[0].estimate()[dim]
            yield e.vertices()[1].estimate()[dim]
            yield None

    def update_graph(self, optimizer, ground_truth=None, covariances=(False, None)):
        """
        1) plots the optimizer's pose, 2) draws an ellipse based on the covariance
        matrix, and 3) plots the covariance points as small red dots.

        Args:
            optimizer (`Optimizer` class instance.): 3D object being posed and is
                used to compute the vertex coordinates for plotting the covariance
                ellipse.
                
                		- `optimizer`: A `Optimizer` object, which contains various
                attributes related to the optimization problem being solved. Some
                of these attributes include:
                		+ `vertex_count`: The number of vertices in the optimizer's graph.
                		+ `vertices`: An array of `Vertex` objects, each representing a
                vertex in the graph.
                		+ `hessian_index`: An array of indices for each vertex's Hessian
                matrix.
                		+ `edges`: An array of edges in the optimizer's graph.
                		+ `graph`: A directed graph object representing the optimizer's
                graph.
                		+ `start_vertices`: An optional list of vertices to start the
                optimization from.
                		+ `stop_vertices`: An optional list of vertices to stop the
                optimization when reached.
                		+ `distance_matrix`: An optional array of distance matrices for
                each vertex pair in the graph.
                		- The `optimizer` object has several methods that can be used
                to interact with its properties and perform various operations,
                such as:
                		+ `optimize()`: Performs a single optimization step using the
                current state of the optimizer.
                		+ `iterations()`: Returns the number of iterations performed so
                far by the optimizer.
                		+ `vertex_count()`: Returns the number of vertices in the
                optimizer's graph.
                		+ `vertices()`: Returns an array of `Vertex` objects representing
                the vertices in the optimizer's graph.
                		+ `hessian_index()`: Returns an array of indices for each vertex's
                Hessian matrix.
                		+ `edges()`: Returns an array of edges in the optimizer's graph.
                		+ `graph()`: Returns a directed graph object representing the
                optimizer's graph.
                		+ `start_vertices()`: Sets or gets the list of vertices to start
                the optimization from.
                		+ `stop_vertices()`: Sets or gets the list of vertices to stop
                the optimization when reached.
                		+ `distance_matrix()`: Sets or gets the array of distance matrices
                for each vertex pair in the graph.
            ground_truth (list): 3D pose of the object or person being analyzed
                in the scene, which is used as a reference to calculate the
                covariance ellipse and its orientation based on the given camera
                parameters and optimizer output.
            covariances (ndarray, specifically an array containing a square matrix
                with two or three elements each, representing a covariance matrix
                as defined by the Optimization module's `Covariance` class in
                scikit-optimize.): 2x2 covariance matrix of the robot's state,
                which is used to compute the eigenvectors and eigenvalues that are
                used to draw the ellipse representing the robot's uncertainty in
                position and orientation.
                
                	1/ `covariances`: This is the input tensor with shape `(N, 3,
                3)`, where `N` is the number of data points, and each column
                represents a covariance matrix between two dimensions of a
                multidimensional dataset. Each element in the matrix is a measure
                of the relationship between two variables.
                	2/ `block`: This attribute allows us to access specific parts of
                the covariance matrix. For example, `block(v.hessian_index(),
                v.hessian_index())` returns a block of the matrix corresponding
                to the last optimizer vertex `v`.
                	3/ ` eigenvalues`: This attribute provides the eigenvectors and
                eigenvalues of the covariance matrix. However, in this implementation,
                it is not used directly.
                	4/ `vertices`: This attribute provides access to the vertices of
                the optimizer object. The last vertex is used to compute the
                rotation matrix.
                	5/ `estimate`: This attribute provides an estimate of the current
                position of the optimizer. It can be used to compute the offset
                of the center of mass from the origin.
                	6/ `hessian_index`: This attribute provides access to the Hessian
                matrix of the objective function at the current vertex. It is used
                to compute the rotation matrix.
                	7/ `lambda_`: This attribute provides the eigenvalues of the
                covariance matrix.
                	8/ `v`: This attribute provides the last optimizer vertex.
                	9/ `T`: This attribute provides the trace of the covariance matrix.
                	10/ `h`: This attribute provides the square root of the determinant
                of the covariance matrix. It is used to solve the characteristic
                polynomial using the p-q formula.
                
                	These properties and attributes are used in the function to compute
                the rotation matrix, draw the ellipse representing the covariance,
                and plot the points from the `points_for_cov` function.

        """
        self.ax.clear()
        # Fijar límites de los ejes
        self.ax.set_xlim([-4000, 4000])  # Por ejemplo, límites para el eje y de 0 a 10
        self.ax.set_ylim([-4000, 4000])  # Por ejemplo, límites para el eje y de 0 a 10
        # Obtener los datos actualizados del optimizador
        edges = optimizer.optimizer.edges()
        vertices = optimizer.optimizer.vertices()

        # edges
        se2_edges = [e for e in edges if type(e) == g2o.EdgeSE2]
        se2_pointxy_edges = [e for e in edges if type(e) == g2o.EdgeSE2PointXY]

        self.ax.plot(list(self.edges_coord(se2_pointxy_edges, 0)), list(self.edges_coord(se2_pointxy_edges, 1)),
                color='firebrick', linestyle='--', linewidth=1, label="Measurement edges")


        # poses of the vertices
        poses = [v.estimate() for v in vertices.values() if type(v) == g2o.VertexSE2]
        measurements = [v.estimate() for v in vertices.values() if type(v) == g2o.VertexPointXY]

        for v in poses:
            self.ax.plot(v.translation()[0], v.translation()[1], 'o', color='lightskyblue')
            self.ax.plot([v.translation()[0], v.translation()[0] + 250 * np.sin(-v.rotation().angle())],
                    [v.translation()[1], v.translation()[1] + 250 * np.cos(-v.rotation().angle())], 'r-', color='green')

        # self.ax.plot([v[0] for v in poses], [v[1] for v in poses], 'o', color='lightskyblue', markersize=10,
        #         label="Poses")
        # Draw arrows for the pose angles

        if ground_truth is not None:
            self.ax.plot([ground_truth[0]], [ground_truth[1]], 'o', color='green', markersize=6,
                    label="Ground truth")
        self.ax.plot([v[0] for v in measurements], [v[1] for v in measurements], '*', color='firebrick',
                markersize=15, label="Measurements")
        # self.ax.plot(list(self.edges_coord(se2_edges, 0)), list(self.edges_coord(se2_edges, 1)),
        #         color='midnightblue', linewidth=1, label="Pose edges")

        if covariances[0]:

            def points_for_cov(cov):
                """
                Generates a set of rotation matrices based on the eigenvalues and
                eigenvectors of a covariance matrix provided as input.

                Args:
                    cov (2D numpy array.): 2x2 covariance matrix of a Gaussian
                        distribution, which is used to calculate the eigen-values
                        and eigen-vectors of the matrix.
                        
                        		- `cov[0, 0]`: The element at row 0 and column 0 of the
                        covariance matrix, denoted by `a`.
                        		- `cov[0, 1]`: The element at row 0 and column 1 of the
                        covariance matrix, denoted by `b`.
                        		- `cov[1, 1]`: The element at row 1 and column 1 of the
                        covariance matrix, denoted by `d`.
                        		- `D`: The determinant of the matrix `a * d - b * b`,
                        denoted by `D`.
                        		- `T`: The trace of the matrix `a + d`, denoted by `T`.
                        		- `h`: A scalar value computed as the square root of
                        0.25 times the product of `T` and `D` minus `h`.
                        		- `lambda1`: A scalar value computed as 0.5 times `T`
                        plus `h`.
                        		- `lambda2`: A scalar value computed as 0.5 times `T`
                        minus `h`.
                        		- `theta`: An angle value computed as the arctangent of
                        `2.0 * b` divided by `a - d`.
                        		- `rotation_matrix`: A numpy array representing a rotation
                        matrix, which is constructed based on the values of
                        `majorAxis`, `minorAxis`, and `alpha`.

                """
                a = cov[0, 0]
                b = cov[0, 1]
                d = cov[1, 1]

                # get eigen-values
                D = a * d - b * b  # determinant of the matrix
                T = a + d  # Trace of the matrix
                h = math.sqrt(0.25 * (T * T) - D)
                lambda1 = 0.5 * T + h  # solving characteristic polynom using p-q-formula
                lambda2 = 0.5 * T - h

                theta = 0.5 * math.atan2(2.0 * b, a - d)
                rotation_matrix = np.array(
                    [[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]]
                )
                majorAxis = 3.0 * math.sqrt(lambda1)
                minorAxis = 3.0 * math.sqrt(lambda2)
                for alpha in np.linspace(0, math.tau, 32):
                    yield np.matmul(
                        rotation_matrix,
                        [
                            majorAxis * math.cos(alpha),
                            minorAxis * math.sin(alpha),
                        ],
                    )

            cov_points_x = []
            cov_points_y = []
            v = optimizer.optimizer.vertices()[optimizer.vertex_count - 1]
            matrix = covariances[1].block(v.hessian_index(), v.hessian_index())
            vertex_offset = v.estimate().to_vector()
            for p in points_for_cov(matrix):
                cov_points_x.append(vertex_offset[0] + p[0])
                cov_points_y.append(vertex_offset[1] + p[1])
            cov_points_x.append(None)
            cov_points_y.append(None)
            # Draw covariance ellipse
            self.ax.plot(cov_points_x, cov_points_y,
                    color='black', linewidth=1, label="Covariance")


        # if covariances[0]:
        #     def draw_covariance_ellipse(x, y, cov):
        #         """ Draw an ellipse based on the covariance matrix. """
        #         lambda_, v = np.linalg.eig(cov)
        #         lambda_ = np.sqrt(lambda_)
        #         print("Pose", x, y)
        #         print("MATRIX", lambda_, v)
        #         ellipse = Ellipse(xy=(x, y), width=lambda_[0] * 2, height=lambda_[1] * 2,
        #                           angle=np.degrees(np.arccos(v[0, 0])), edgecolor='green', facecolor='none')
        #         self.ax.add_patch(ellipse)
        #     v = optimizer.optimizer.vertices()[optimizer.vertex_count - 1]
        #
        #
        #     matrix = covariances[1].block(v.hessian_index(), v.hessian_index())
        #
        #     vertex_offset = v.estimate().to_vector()
        #     draw_covariance_ellipse(vertex_offset[0], vertex_offset[1], matrix)


            # def points_for_cov(cov):
            #     a = cov[0, 0]
            #     b = cov[0, 1]
            #     d = cov[1, 1]
            #
            #     # get eigen-values
            #     D = a * d - b * b  # determinant of the matrix
            #     T = a + d  # Trace of the matrix
            #     h = math.sqrt(0.25 * (T * T) - D)
            #     lambda1 = 0.5 * T + h  # solving characteristic polynom using p-q-formula
            #     lambda2 = 0.5 * T - h
            #
            #     theta = 0.5 * math.atan2(2.0 * b, a - d)
            #     rotation_matrix = np.array(
            #         [[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]]
            #     )
            #     majorAxis = 3.0 * math.sqrt(lambda1)
            #     minorAxis = 3.0 * math.sqrt(lambda2)
            #     for alpha in np.linspace(0, math.tau, 32):
            #         yield np.matmul(
            #             rotation_matrix,
            #             [
            #                 majorAxis * math.cos(alpha),
            #                 minorAxis * math.sin(alpha),
            #             ],
            #         )
            #
            # cov_points_x = []
            # cov_points_y = []
            # # Get last optimizer vertex
            # v = optimizer.optimizer.vertices()[optimizer.vertex_count - 1]
            #
            #
            # matrix = covariances[1].block(v.hessian_index(), v.hessian_index())
            # print("MATRIX", matrix)
            # vertex_offset = v.estimate().to_vector()
            # for p in points_for_cov(matrix):
            #     cov_points_x.append(vertex_offset[0] + p[0])
            #     cov_points_y.append(vertex_offset[1] + p[1])
            # print("Covariance points", cov_points_x, cov_points_y)
            # cov_points_x.append(None)
            # cov_points_y.append(None)
            #
            # self.ax.plot(cov_points_x, cov_points_y,
            #         color='black', linewidth=5, label="Covariance")


        self.ax.legend()
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
