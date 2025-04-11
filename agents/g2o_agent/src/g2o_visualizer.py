import matplotlib.pyplot as plt
import g2o
import time
import numpy as np
from matplotlib.patches import Polygon, Ellipse, Circle
import math

class G2OVisualizer:
    def __init__(self, title):
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
        for e in edges:
            yield e.vertices()[0].estimate()[dim]
            yield e.vertices()[1].estimate()[dim]
            yield None

    def update_graph(self, optimizer, ground_truth=None, covariances=(False, None)):
        self.ax.clear()
        # Fijar límites de los ejes
        self.ax.set_xlim([-8000, 8000])  # Por ejemplo, límites para el eje y de 0 a 10
        self.ax.set_ylim([-8000, 8000])  # Por ejemplo, límites para el eje y de 0 a 10
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
