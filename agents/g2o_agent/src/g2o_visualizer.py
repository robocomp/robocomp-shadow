import matplotlib.pyplot as plt
import g2o
import time
import numpy as np

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

    def update_graph(self, optimizer, ground_truth=None):
        self.ax.clear()
        # Fijar límites de los ejes
        # self.ax.set_xlim([-3000, 3000])  # Por ejemplo, límites para el eje x de 0 a 10
        # self.ax.set_ylim([-3000, 3000])  # Por ejemplo, límites para el eje y de 0 a 10
        # Obtener los datos actualizados del optimizador
        edges = optimizer.edges()
        vertices = optimizer.vertices()

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
        self.ax.plot(list(self.edges_coord(se2_edges, 0)), list(self.edges_coord(se2_edges, 1)),
                color='midnightblue', linewidth=1, label="Pose edges")
        self.ax.legend()
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
