#!/usr/bin/python3
# -*- coding: utf-8 -*-
#
#    Copyright (C) 2023 by YOUR NAME HERE
#
#    This file is part of RoboComp
#
#    RoboComp is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    RoboComp is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with RoboComp.  If not, see <http://www.gnu.org/licenses/>.
#
import traceback

from PySide2.QtCore import QTimer
from PySide2.QtWidgets import QApplication
from rich.console import Console
from genericworker import *
import interfaces as ifaces
import numpy as np
import cv2
import time

sys.path.append('/opt/robocomp/lib')
console = Console(highlight=False)
sys.path.append('/home/robocomp/robocomp/components/robocomp-shadow/insect/byte_tracker/ByteTrack')
# from yolox.tracker.byte_tracker_depth import BYTETracker as BYTETrackerDepth
from yolox.tracker.byte_tracker import BYTETracker
from dataclasses import dataclass

class SpecificWorker(GenericWorker):
    def __init__(self, proxy_map, startup_check=False):
        super(SpecificWorker, self).__init__(proxy_map)
        self.Period = 50

        if startup_check:
            self.startup_check()
        else:
            self.lidar_in_image = None

            # ROI parameters. Must be filled up
            self.final_xsize = 0
            self.final_ysize = 0
            self.roi_xcenter = 512
            self.roi_ycenter = 256
            self.roi_xsize = 512
            self.roi_ysize = 256
            self.original_fov = 360
            # COPPELIA LIDAR
            self.cam_to_lidar = self.make_matrix_rt(0, 0, 0, 0, 0,
                                                    108.51)
            self.lidar_to_cam = np.linalg.inv(self.cam_to_lidar)
            self.lidar_to_cams = {"cam_front": self.make_matrix_rt(0, 0, 0, 0, 0, -108.51),
                                  "cam_right": self.make_matrix_rt(0, 0, np.pi / 2, 0, 0, -108.51),
                                  "cam_back_1": self.make_matrix_rt(0, 0, np.pi, 0, 0, -108.51),
                                  "cam_back_2": self.make_matrix_rt(0, 0, np.pi, 0, 0, -108.51),
                                  "cam_left": self.make_matrix_rt(0, 0, -np.pi / 2, 0, 0, -108.51)}
            #REAL LIDAR
            # self.overlap = 260
            self.overlap = 0
            self.rvec = np.array( [np.deg2rad(90), np.deg2rad(0), np.deg2rad(0)])
            self.tvec = np.array([0., 220., 30.])
            self.rvec_back = np.array([np.deg2rad(-90), np.deg2rad(0), np.deg2rad(0)])
            self.tvec_back = np.array([0., -220., 0.])
            self.distance_threshold = 7000

            self.aperture = np.deg2rad(180)



            self.K = np.array([[1080, 0.0, 1824], [0.0, 1080, 1824],
                               [0.0, 0.0, 1.0]])

            self.D = np.array([[0.32508720224831864], [-0.396793518453425], [0.20325216832157427], [-0.03725173372715627]])
            # self.D = np.array([[0.0], [0.0], [0.0], [0.0]])
            self.fisheye_calibration_size = (3648, 3648)


            self.focal_x = 128
            self.focal_y = 128
            self.width_img = 1024
            self.height_img = 512
            self.conditions = [
                ((-np.pi / 4, np.pi / 4), "cam_front"),
                ((np.pi / 4, (np.pi * 3) / 4), "cam_right"),
                ((-(np.pi * 3) / 4, -np.pi / 4), "cam_left"),
                ((-np.pi, -(np.pi * 3) / 4), "cam_back_2"),
                (((np.pi * 3) / 4, np.pi), "cam_back_1"),
            ]
            self.conditions_real = [
                ((-np.pi / 2, np.pi / 2), "cam_front"),
                ((-np.pi / 2, np.pi / 2), "cam_back"),
            ]

            # ROI offsets respect the original image
            self.x_roi_offset = 0
            self.y_roi_offset = 0
            self.objects_read = []
            self.objects_write = []
            self.display = False
            self.simulator = False

            # Hz
            self.cont = 1
            self.last_time = time.time()
            self.fps = 0

            # read test image to get sizes
            started_camera = False
            init_time = time.time()
            while not started_camera and (time.time() - init_time) < 10:
                try:
                    rgb = self.camera360rgb_proxy.getROI(-1, -1, -1, -1, -1, -1)
                    # self.center_x = rgb.width // 2
                    # self.center_y = rgb.height // 2
                    print("Camera specs:")
                    print(" width:", rgb.width)
                    print(" height:", rgb.height)
                    print(" depth", rgb.depth)
                    print(" focalx", rgb.focalx)
                    print(" focaly", rgb.focaly)
                    print(" period", rgb.period)
                    print(" ratio {:.2f}".format(rgb.width/rgb.height))

                    started_camera = True
                except Ice.Exception as e:
                    traceback.print_exc()
                    print(e, "Trying again...")
                    time.sleep(1)

            self.timer.timeout.connect(self.compute)
            self.timer.start(self.Period)

    def __del__(self):
        """Destructor"""

    def setParams(self, params):
        try:
            self.tracker = BYTETracker(frame_rate=30)
            if params["display"] == "true" or params["display"] == "True":
                self.display = True
            if params["simulator"] == "true" or params["coppelia"] == "True":
                self.simulator = True
        except:
            traceback.print_exc()
            print("Error reading config params")

        return True


    @QtCore.Slot()
    def compute(self):
        # project lidar points on image
        self.lidar_in_image = self.lidar_points()
        # Get tracks from Bytetrack and convert data to VisualElements interface
        if self.display:
            img = self.display_data(self.read_image())
            if img is not None:
                cv2.imshow("Image", img)
                cv2.waitKey(1)
        self.show_fps()

    def lidar_points(self):
        """
        This method interacts with a LIDAR device, retrieves 3D data points within a specific region of interest (ROI)
        and projects them onto an image plane. Each LIDAR data point is represented in a 3D space as [x, y, z].

        The function works in the following way:

        1. It calculates the start and end angles for the ROI by calling self.calculate_roi_angles() method.
        2. Using the LIDAR device interface (self.lidar3d_proxy), it retrieves the LIDAR data points that fall
           within the specified range of angles.
        3. If there is an error in fetching data from the LIDAR device, it prints the error and the stack trace.
        4. The raw LIDAR data points are converted into a numpy array for further processing.
        5. If the application is set to simulator mode, the points are transformed using the self.lidar_coppelia()
           method, else they are transformed using self.lidar_real() method.
        6. The transformed LIDAR points are returned as a numpy array.

        Returns:
        --------
        numpy array
            The return value is a numpy array of projected LIDAR points that fall within the image bounds.
            Each point is represented as a 3D point [x, y, distance].

        Exceptions:
        -----------
        Ice.Exception
            Exceptions from the Ice framework (used for LIDAR data retrieval) are caught and printed to the standard error.
        """

        # Calculate angles for the region of interest.
        start_angle, len_angle = self.calculate_roi_angles_coppelia() if self.simulator else self.calculate_roi_angles_real()

        # Fetch LIDAR points that fall within the specified angle range.

        try:
            points = self.lidar3d_proxy.getLidarData(int(start_angle), int(len_angle))

            #points = self.lidar3d_proxy.getLidarData(270, 900)
            # print(points)
        except Ice.Exception as e:
            traceback.print_exc()
            print(e, "Error connecting to Lidar3D")

        # Convert points into a numpy array and scale from millimeters to meters.
        points_array = np.array([[p.x, p.y, p.z] for p in points])
        return np.array(self.lidar_coppelia(points_array)) if self.simulator else np.array(self.lidar_real(points_array))

    def lidar_coppelia(self, points_array):
        """
        Method for projecting LIDAR points onto the image plane considering multiple camera views
        in a simulation environment like CoppeliaSim.

        Args:
            points_array (numpy array): A numpy array containing the LIDAR points to be projected.
        """

        # Initialize list to hold LIDAR points projected onto the image plane.
        lidar_in_image = []
        # Define camera x and y offsets for each camera view.
        half_width = self.width_img / 2
        quarter_width = half_width / 2
        half_height = self.height_img / 2
        quarter_height = half_height / 4
        cx_offsets = {
            "cam_front": half_width,
            "cam_right": half_width + quarter_width,
            "cam_left": half_width - quarter_width,
            "cam_back_1": self.width_img,
            "cam_back_2": 0
        }
        cy_offsets = {
            "cam_front": half_height,
            "cam_right": half_height + quarter_height,
            "cam_left": half_height + quarter_height,
            "cam_back_1": half_height,
            "cam_back_2": half_height
        }

        points_array = np.append(points_array, np.ones((points_array.shape[0], 1)), axis=1)

        # Calculate the angle for each point.
        angles = np.arctan2(points_array[:, 0], points_array[:, 1])
        # Define a small value to avoid division by zero.
        epsilon = 1e-7
        for condition, cam in self.conditions:
            # Identify points that fall within the current camera view based on angle.
            indices = np.where((condition[0] < angles) & (angles <= condition[1]))[0]

            if len(indices) > 0:
                transformation_matrix = self.lidar_to_cams[cam].T
                cx = cx_offsets[cam]
                cy = cy_offsets[cam]

                # Apply transformation matrix to points.
                transformed_points = np.dot(points_array[indices], transformation_matrix)
                transformed_points[:, 1] += epsilon  # Add epsilon to y coordinate to avoid division by zero.

                # Project points onto the image plane.
                x = (self.focal_x * transformed_points[:, 0] / transformed_points[:, 1]) + cx
                y = (-self.focal_y * transformed_points[:, 2] / transformed_points[:, 1]) + cy
                transformed_points = np.dot(transformed_points, np.linalg.inv(transformation_matrix))
                # Identify points that fall within the image bounds.
                valid_indices = np.where((0 <= x) & (x < self.width_img) & (0 <= y) & (y < self.height_img))[0]
                # Add valid points to the lidar_in_image list.
                if len(valid_indices) > 0:
                    lidar_in_image.extend(np.column_stack([x[valid_indices],
                                                           y[valid_indices],
                                                           np.linalg.norm(transformed_points[valid_indices, :3],
                                                                          axis=1),
                                                           transformed_points[valid_indices, 0],
                                                           transformed_points[valid_indices, 1],
                                                           transformed_points[valid_indices, 2]]).tolist())
        return lidar_in_image

    def lidar_real(self, lidar_points):
        """
        Method for projecting real-world LIDAR points onto the image plane considering multiple camera views.

        Args:
            lidar_points (numpy array): A numpy array containing the real-world LIDAR points to be projected.
        """

        # Cast the lidar_points to float data type.
        lidar_points = lidar_points.astype("f")

        # Split the points into front and back groups based on the 'overlap' parameter.
        lidar_front = lidar_points[lidar_points[:, 1] > -self.overlap]
        lidar_back = lidar_points[lidar_points[:, 1] < self.overlap]

        # Initialize list to hold LIDAR points projected onto the image plane.
        lidar_in_image = []

        # For each condition and camera in the real-world conditions:
        for condition, cam in self.conditions_real:
            # If the current camera is the front camera:
            if cam == "cam_front":
                # Project the points onto the image plane using the 'proyect_front_real_lidar' method.
                transformed_points = self.proyect_front_real_lidar(lidar_front.reshape(-1, 1, 3))
                # Set the points array to lidar_front.
                points_array = lidar_front
            else:  # If the current camera is not the front camera:
                # Project the points onto the image plane using the 'proyect_back_real_lidar' method.
                transformed_points = self.proyect_back_real_lidar(lidar_back.reshape(-1, 1, 3))
                # Set the points array to lidar_back.
                points_array = lidar_back

            # Add the transformed points and their respective information to the lidar_in_image list.
            lidar_in_image.extend(np.column_stack([transformed_points[:, 0],
                                                   transformed_points[:, 1],
                                                   np.linalg.norm(points_array[:, :3], axis=1),
                                                   points_array[:, 0],
                                                   points_array[:, 1],
                                                   points_array[:, 2]
                                                   ]).tolist())
        # Return the list of LIDAR points projected onto the image plane.
        return lidar_in_image

    def proyect_front_real_lidar(self, points):
        """
        Method for projecting real-world LIDAR points onto the image plane for the front camera view using a fisheye lens model.

        Args:
            points (numpy array): A numpy array containing the real-world LIDAR points to be projected.
        """

        # Project the points onto the image plane using the fisheye lens model with the given rotation and translation vectors, camera matrix, and distortion coefficients.
        # The alpha parameter is set to 0 for no scaling.
        front_points_2d, _ = cv2.fisheye.projectPoints(points, self.rvec, self.tvec, self.K, self.D, alpha=0)

        # Convert the fisheye points to equirectangular projection, taking into account the fisheye calibration size and the desired image size.
        equirect_points_front = self.fish2equirect(front_points_2d[:, 0, :], self.fisheye_calibration_size,
                                                   (self.width_img, self.height_img))

        # Adjust the x coordinates of the points by half the image height.
        equirect_points_front[:, 0] += int(self.height_img // 2)

        # Return the projected points.
        return equirect_points_front

    def proyect_back_real_lidar(self, points):
        """
        Method for projecting real-world LIDAR points onto the image plane for the back camera view using a fisheye lens model.

        Args:
            points (numpy array): A numpy array containing the real-world LIDAR points to be projected.
        """
        # Invert the y coordinates.
        points[:, :, 1] = -points[:, :, 1]

        # Project the points onto the image plane using the fisheye lens model with the given rotation and translation vectors, camera matrix, and distortion coefficients for the back camera.
        # The alpha parameter is set to 0 for no scaling.
        back_points_2d, _ = cv2.fisheye.projectPoints(points, self.rvec_back, self.tvec_back, self.K, self.D, alpha=0)

        # Convert the fisheye points to equirectangular projection, taking into account the fisheye calibration size and the desired image size.
        equirect_points_back = self.fish2equirect(back_points_2d[:, 0, :], self.fisheye_calibration_size,
                                                  (self.width_img, self.height_img))

        # Find the indices of the points where the x coordinate is greater than a quarter of the image width.
        greater_index = equirect_points_back[:, 0] > (self.width_img // 4)

        # If the x coordinate is greater than a quarter of the image width, subtract a quarter of the image width.
        equirect_points_back[greater_index, 0] -= (self.width_img // 4)

        # If the x coordinate is less than or equal to a quarter of the image width, add three quarters of the image width.
        equirect_points_back[~greater_index, 0] += (self.width_img * 3) // 4

        # Return the projected points.
        return equirect_points_back

    def fish2equirect(self, points, size_in, size_out):
        """
        Translates points from a fisheye perspective to an equirectangular plane.

        Parameters:
            points (numpy array): Points in the fisheye perspective to be translated. They should be in the format [[x,y],...].
            size_in (list): The size of the fisheye image in the format [x,y].
            size_out (list): The desired size of the output equirectangular image in the format [x,y].

        Returns:
            pointsEquirect (numpy array): Points in the equirectangular plane, in the format [[x,y],...].
        """

        # Dimensions of the fisheye image.
        width = size_in[0]
        height = size_in[1]

        # Desired dimensions of the output equirectangular image.
        dst_width = size_out[0]
        dst_height = size_out[1]

        # Calculate the center of the image.
        center_x = width // 2
        center_y = height // 2

        # Normalize the points to the range [-1,1], with inversion in y.
        src_x_norm = (points[:, 0] - center_x) * 2 / width
        src_y_norm = -(points[:, 1] - center_y) * 2 / height

        # Calculate the radius or hypotenuse.
        r = np.sqrt(src_x_norm ** 2 + src_y_norm ** 2)

        # Convert to 3D fisheye points.
        p_x = src_x_norm
        p_z = src_y_norm
        p_y = np.where(r != 0, r / np.tan(r * self.aperture / 2), 0)

        # Convert 3D fisheye points to 3D lat/lon.
        latitude = np.arctan2(p_z, np.sqrt(p_x ** 2 + p_y ** 2))
        longitude = np.arctan2(p_y, p_x)

        # Convert 3D lat/lon to 2D equirectangular.
        dst_x_norm = longitude / np.pi
        dst_y_norm = latitude * 2 / np.pi

        # Scale the equirectangular points to the range [0, dst_width] and [0, dst_height], with inversion in y.
        x = ((-dst_x_norm / 2 + 0.5) * dst_width).astype(int)
        y = ((-dst_y_norm / 2 + 0.5) * dst_height).astype(int)

        # Return the equirectangular points.
        return np.stack((x, y), axis=-1)

    def make_matrix_rt(self, roll, pitch, heading, x0, y0, z0):
        """
           Constructs a rotation-translation matrix given the roll, pitch, and heading angles,
           along with the x, y, and z coordinates of a translation vector.

           This method generates a 4x4 matrix representing the transformation. It does this by
           first computing the individual rotation matrices for the roll, pitch, and heading angles,
           and then combines these with the translation vector to form a single homogeneous transformation matrix.

           Parameters:
           roll : float
               The roll angle in radians. This represents a rotation around the x-axis.
           pitch : float
               The pitch angle in radians. This represents a rotation around the y-axis.
           heading : float
               The heading angle in radians. This represents a rotation around the z-axis.
           x0, y0, z0 : float
               The x, y, and z coordinates of the translation vector.

           Returns:
           mat : ndarray
               A 4x4 numpy array representing the rotation-translation matrix.

           """
        sa, ca = np.sin(roll), np.cos(roll)
        sb, cb = np.sin(pitch), np.cos(pitch)
        sg, cg = np.sin(heading), np.cos(heading)

        mat = np.array([
            [cb * cg, sa * sb * cg + ca * sg, sa * sg - ca * sb * cg, x0],
            [-cb * sg, ca * cg - sa * sb * sg, sa * cg + ca * sb * sg, y0],
            [sb, -sa * cb, ca * cb, z0],
            [0, 0, 0, 1]
        ])
        return mat


    def calculate_roi_angles_real(self):
        """
        Este método calcula los ángulos de inicio y longitud de un ROI (Región de Interés)
        en una imagen panorámica que cubre 360 grados.

        La imagen se representa como una línea horizontal, y la ROI es una subsección
        de esa línea definida por un centro (self.roi_xcenter) y un tamaño (self.roi_xsize).

        Returns:
            tuple: un par de ángulos (start_angle, len_angle) representando el ángulo
                   de inicio y la longitud del ángulo de la ROI en la imagen, respectivamente.
        """
        # La imagen original cubre 360 grados
        original_fov = 360

        # Calcula las proporciones de inicio y fin de la ROI con respecto al ancho de la imagen.
        # Estas proporciones representan la fracción del ancho de la imagen que cubre la ROI.
        # print("roixsize: ", self.roi_xsize)
        # print("roixcenter: ", self.roi_xcenter)
        # print("WIDHT IMAGE ", self.width_img)
        if self.roi_xcenter - self.roi_xsize // 2 < 0:
            print("MENOR 0")
            start_ratio = ((self.roi_xcenter - self.roi_xsize // 2)+self.width_img) / self.width_img
        else:
            start_ratio = (self.roi_xcenter - self.roi_xsize // 2) / self.width_img

        if (self.roi_xcenter + self.roi_xsize // 2) > self.width_img:
            print("MAYOR 0")
            end_ratio = (self.roi_xcenter + self.roi_xsize // 2- self.width_img) / self.width_img
        else:
            end_ratio = (self.roi_xcenter + self.roi_xsize // 2) / self.width_img

        # Calcula los ángulos de inicio y fin de la ROI.
        # Primero, multiplica las proporciones de inicio y fin por el campo de visión original (original_fov)
        # para obtener los ángulos en el rango de la imagen completa.
        start_angle = original_fov * start_ratio
        end_angle = original_fov * end_ratio

        # if start_angle < 0:
        #     start_angle += 360
        # if end_angle < 0:
        #     end_angle += 360
        # Calcula la longitud del ángulo de la ROI, que es la diferencia absoluta entre
        # los ángulos de inicio y fin.
        len_angle = abs(end_angle - start_angle)

        # Devuelve los ángulos de inicio y longitud.
        return start_angle, len_angle

    def calculate_roi_angles_coppelia(self):
        """
        Este método calcula los ángulos de inicio y longitud de un ROI (Región de Interés)
        en una imagen panorámica que cubre 360 grados.

        La imagen se representa como una línea horizontal, y la ROI es una subsección
        de esa línea definida por un centro (self.roi_xcenter) y un tamaño (self.roi_xsize).

        Los ángulos calculados se mapean a un rango de 0 a 900 grados,
        donde 0 corresponde a 0 grados, 450 a 180 grados, y 900 a -180 grados.

        Returns:
            tuple: un par de ángulos (start_angle, len_angle) representando el ángulo
                   de inicio y la longitud del ángulo de la ROI en la imagen, respectivamente.
        """
        # La imagen original cubre 360 grados
        original_fov = 360

        # Calcula las proporciones de inicio y fin de la ROI con respecto al ancho de la imagen.
        # Estas proporciones representan la fracción del ancho de la imagen que cubre la ROI.
        start_ratio = (self.roi_xcenter - self.roi_xsize // 2) / self.width_img
        end_ratio = (self.roi_xcenter + self.roi_xsize // 2) / self.width_img

        # Calcula los ángulos de inicio y fin de la ROI.
        # Primero, multiplica las proporciones de inicio y fin por el campo de visión original (original_fov)
        # para obtener los ángulos en el rango de la imagen completa.
        # Luego, resta 180 para convertirlos al rango [-180, 180].
        # Finalmente, llama a remap_angle para convertir estos ángulos al rango [0, 900] como se describió anteriormente.
        start_angle = self.remap_angle(original_fov * start_ratio - 180)

        end_angle = self.remap_angle(original_fov * end_ratio - 180)


        # Calcula la longitud del ángulo de la ROI, que es la diferencia absoluta entre
        # los ángulos de inicio y fin.
        len_angle = abs(end_angle - start_angle)

        # Devuelve los ángulos de inicio y longitud.
        return start_angle, len_angle

    def remap_angle(self, angle):
        if angle >= 0:
            return 2.5 * angle
        else:
            return 900 + 2.5 * angle
    def read_image(self):
        """
        Retrieves an image from a 360-degree RGB camera and reshapes it into the proper format.

        The method uses the camera360rgb_proxy object to call its getROI method, which gets a Region of
        Interest (ROI) from the 360-degree camera. In this case, the whole image is retrieved as ROI
        dimensions are set to -1. The image is then reshaped from a 1D array into a 3D array representing
        height, width, and color channels of the image. The image dimensions are also stored in the
        instance variables width_img and height_img.

        Returns:
        rgb_frame : numpy array
            The image data in a 3D numpy array with shape (height, width, 3). Each element of the array
            represents a pixel in the image and its corresponding RGB color values.
        """

        # Retrieve the image data from the 360-degree camera.
        try:
            rgb = self.camera360rgb_proxy.getROI(-1, -1, -1, -1, -1, -1)
        except Ice.Exception as e:
            traceback.print_exc()
            print(e, "Error connecting to Camera360")

        # Reshape the 1D array of pixel data into a 3D array representing the RGB image.
        rgb_frame = np.frombuffer(rgb.image, dtype=np.uint8).reshape((rgb.height, rgb.width, 3))

        # Store the image dimensions for later use.
        self.width_img = rgb.width
        self.height_img = rgb.height

        return rgb_frame

    ####################################################################################################
    ####################################################################################################
    ### Methods called from Ice interface
    ####################################################################################################
    def process_visual_objects(self, visual_objects):
        """
           Extracts and organizes information about visual objects detected in an image.

           This method processes a list of visual objects, each represented by a data structure containing
           information such as the object's score (a measure of the confidence in the detection), bounding
           box coordinates, type, and region of interest (ROI). It organizes this information into a dictionary
           and also updates the properties of the class instance related to the ROI.

           Parameters:
           visual_objects : list
               A list of visual objects, where each object contains information about a detection in an image.

           Returns:
           data : dict
               A dictionary containing the following keys:
               - 'scores': A list of scores for each object, indicating the confidence of the detection.
               - 'boxes': A list of bounding boxes for each object, each represented by a list of four
                 coordinates: left, top, right, bottom.
               - 'clases': A list of the types of each object.
               - 'roi': A list of the ROI for each object.

           If an exception occurs while processing the visual objects, this method will print an error message
           and the stack trace, and return an empty dictionary.

           Raises:
           Ice.Exception : If an error occurs while reading from the Visual Objects interface.
           """
        try:
            data = {
                "scores": [object.score for object in visual_objects],
                "boxes": [[object.left, object.top, object.right, object.bot] for object in visual_objects],
                "clases": [object.type for object in visual_objects],
                "roi": [object.roi for object in visual_objects]
            }

            # get roi params from first visual object since all are the same
            if visual_objects:
                roi = visual_objects[0].roi
                self.final_xsize = roi.finalxsize
                self.final_ysize = roi.finalysize
                self.roi_xcenter = roi.xcenter
                self.roi_ycenter = roi.ycenter
                self.roi_xsize = roi.xsize
                self.roi_ysize = roi.ysize
                self.x_roi_offset = self.roi_xcenter - self.roi_xsize / 2
                self.y_roi_offset = self.roi_ycenter - self.roi_ysize / 2
                self.x_factor = self.roi_xsize / self.final_xsize
                self.y_factor = self.roi_ysize / self.final_ysize

        except Ice.Exception as e:
            traceback.print_exc()
            print(e, "Error reading from Visual Objects interface")
        return data

    def to_visualelements_interface(self, tracks):
        targets = [
                ifaces.RoboCompVisualElements.TObject(
                roi=track.roi, id=int(track.track_id), score=track.score,
                left=int(track.tlwh[0]), top=int(track.tlwh[1]),
                right=int(track.tlwh[0] + track.tlwh[2]),
                bot=int(track.tlwh[1] + track.tlwh[3]), type=track.clase,
            )
            for track in tracks
        ]
        objects = self.distance_to_object(targets)
        self.objects_write = ifaces.RoboCompVisualElements.TObjects(objects)

        # swap
        self.objects_write, self.objects_read = self.objects_read, self.objects_write
        return self.objects_read


    def distance_to_object(self, objects):

        if len(self.objects) == 0:
            return []

        # Calculate scaling factors and offsets based on the ROI dimensions.
        x_factor = self.roi_xsize / self.final_xsize
        y_factor = self.roi_ysize / self.final_ysize
        x_offset = self.roi_xcenter - self.roi_xsize / 2
        y_offset = self.roi_ycenter - self.roi_ysize / 2
        # Iterate over all objects.
        for element in objects:
            # Calculate the coordinates of the bounding rectangle for each object.
            x0 = int(element.left * x_factor + x_offset)
            y0 = int(element.top * y_factor + y_offset)
            x1 = int(element.right * x_factor + x_offset)
            y1 = int(element.bot * y_factor + y_offset)
            if self.lidar_in_image is not None:
                _, centroid, depth = self.points_in_bbox(self.lidar_in_image, x0, y0, x1, y1)
                if depth != None:
                    element.depth = depth
                    element.x = centroid[0]
                    element.y = centroid[1]
                    element.z = centroid[2]
            else:
                print("Warning, no lidar data yet")

        return objects

    def points_in_bbox(self, points, x1, y1, x2, y2):
        """
        Searches within a list of points for those located within a bounding box and finds the minimum
        distance among these points.

        Parameters:
        points : list
            A list of tuples, where each tuple represents a point and consists of (x, y, distance, xw, yw, zw).
        x1, y1, x2, y2 : float
            The coordinates defining the bounding box. (x1, y1) represents the lower-left corner and
            (x2, y2) the upper-right corner.

        Returns:
        in_box_points: array of all points in bbox
        centroid : point
            x, y coordinates of the centroid of the in_box_points
        min_distance : float
            The minimum distance among the points within the bounding box. If there are no points
            within the box, returns infinity.
        """

        # if x2 > self.width_img:
        #     x2 -= self.width_img
        # if x2 < 0:
        #     x2 += self.width_img
        if x1 < 0:
            x1 += self.width_img
        if x2 < 0:
            x2 = -x2


        # Usual case
        if x1 < x2:
            in_box_points = [p for p in points if x1 <= p[0] <= x2 and y1 <= p[1] <= y2]
        else:
            in_box_points = [p for p in points if ((x1 <= p[0] < self.width_img) or (0 < p[0] <= x2)) and y1 <= p[1] <= y2]
        min_distance = min((p[2] for p in in_box_points), default=float('inf'))

        if len(in_box_points) > 0:
            x = [p[3] for p in in_box_points]  # X  get middle 3D point en X,Z plane
            z = [p[5] for p in in_box_points]  # Z
            y = min((p[4] for p in in_box_points), default=float('inf')) #MIN Y
            #centroid = (sum(x) / len(in_box_points), -sum(z) / len(in_box_points))
            centroid = (sum(x) / len(in_box_points), y, sum(z) / len(in_box_points) )
            return in_box_points, centroid, min_distance
        else:
            return in_box_points, (), None
    def display_data(self, image):
        """
        Displays data on an image, including objects of interest and corresponding LIDAR points.
        Each object is outlined by a rectangle, and each LIDAR point is represented as a green circle.
        Information about the object's type, detection score, ID, and the minimum distance to the LIDAR points
        is displayed above each rectangle.

        Parameters:
        image : numpy array
            The image on which to display the data.

        Returns:
        image : numpy array
            The original image, modified to include visualizations of the object and LIDAR data.
        """
        for i in self.lidar_in_image:
            cv2.circle(image, (int(i[0]), int(i[1])), 1, (0, 255, 0), 1)
        # Check if there are any objects to display.
        if len(self.objects) == 0:
            return image

        # Set the font for displaying text on the image.
        font = cv2.FONT_HERSHEY_SIMPLEX

        # Iterate over all objects.
        for element in self.objects:
            # Define the ROI (region of interest) for each object.
            roi = element.roi

            # Calculate scaling factors and offsets based on the ROI dimensions.
            x_factor = roi.xsize / roi.finalxsize
            y_factor = roi.ysize / roi.finalysize
            x_offset = roi.xcenter - roi.xsize / 2
            y_offset = roi.ycenter - roi.ysize / 2

            # Calculate the coordinates of the bounding rectangle for each object.
            x0 = int(element.left * x_factor + x_offset)
            y0 = int(element.top * y_factor + y_offset)
            x1 = int(element.right * x_factor + x_offset)
            y1 = int(element.bot * y_factor + y_offset)

            # Draw the bounding rectangle on the image.
            cv2.rectangle(image, (x0, y0), (x1, y1), (0, 255, 0), 2)


            # Identify the LIDAR points within the bounding rectangle and draw them on the image.
            results, centroid, min_distance = self.points_in_bbox(self.lidar_in_image, x0, y0, x1, y1)
            for i in results:
                cv2.circle(image, (int(i[0]), int(i[1])), 1, (0, 255, 0), 1)

            # Prepare the text for each object.
            text = 'Class: {} - Score: {:.1f}% - ID: {} -Dist: {}'.format(element.type, element.score * 100, element.id,
                                                                          min_distance/1000 )
            # Calculate the size of the text.
            txt_size = cv2.getTextSize(text, font, 0.4, 1)[0]

            # Draw a filled rectangle as the background of the text.
            cv2.rectangle(
                image,
                (x0, y0 + 1),
                (x0 + txt_size[0] + 1, y0 + int(1.5 * txt_size[1])),
                (255, 0, 0),
                -1
            )
            # # Put the text on the image.
            cv2.putText(image, text, (x0, y0 + txt_size[1]), font, 0.4, (0, 255, 0), thickness=1)

        # Return the image with the drawn objects and LIDAR points.
        return image

    def show_fps(self):
        """
        Calculates and displays the frames per second (FPS) information for the processing.

        This method uses a simple timer to measure the time that elapses between successive calls to this
        method, and counts the number of frames processed during that time. If more than one second has
        elapsed since the last time this method was called, it will print the number of frames processed
        in that second (i.e., the FPS) and the period of each frame in milliseconds. The period is the
        reciprocal of the FPS and represents the time taken to process each frame. The count is then reset
        to zero. If less than one second has passed since the last call, it simply increments the count of
        frames.

        Note:
        This method does not return any value. It simply prints the FPS and frame period to the console.
        """

        # If more than one second has elapsed since the last time this method was called.
        if time.time() - self.last_time > 1:
            self.last_time = time.time()
            cur_period = int(1000. / self.cont)  # Calculate the frame period in milliseconds.

            # Print the FPS (i.e., the frame count) and the frame period.
            print("Freq:", self.cont, "ms. Curr period:", cur_period)

            self.cont = 0  # Reset the frame count.

        # If less than one second has passed since the last call, increment the frame count.
        else:
            self.cont += 1

    ##############################################################################################

    def startup_check(self):
        print(f"Testing RoboCompYoloObjects.TBox from ifaces.RoboCompYoloObjects")
        test = ifaces.RoboCompYoloObjects.TBox()
        print(f"Testing RoboCompYoloObjects.TKeyPoint from ifaces.RoboCompYoloObjects")
        test = ifaces.RoboCompYoloObjects.TKeyPoint()
        print(f"Testing RoboCompYoloObjects.TPerson from ifaces.RoboCompYoloObjects")
        test = ifaces.RoboCompYoloObjects.TPerson()
        print(f"Testing RoboCompYoloObjects.TConnection from ifaces.RoboCompYoloObjects")
        test = ifaces.RoboCompYoloObjects.TConnection()
        print(f"Testing RoboCompYoloObjects.TJointData from ifaces.RoboCompYoloObjects")
        test = ifaces.RoboCompYoloObjects.TJointData()
        print(f"Testing RoboCompYoloObjects.TData from ifaces.RoboCompYoloObjects")
        test = ifaces.RoboCompYoloObjects.TData()
        QTimer.singleShot(200, QApplication.instance().quit)

    ##############################################################################################
    # IMPLEMENTATION of getVisualObjects method from VisualElements interface
    ##############################################################################################

    def VisualElements_getVisualObjects(self, objects):
        # Read visual elements from segmentator
        data = self.process_visual_objects(objects)
        self.objects = objects
        # Get tracks from Bytetrack and convert data to VisualElements interface
        tracks = self.tracker.update_original(np.array(data["scores"]),
                                                       np.array(data["boxes"]),
                                                       np.array(data["clases"]),
                                                       np.array(data["roi"]))
        return self.to_visualelements_interface(tracks)
    ##############################################################################################


    # ===================================================================
    # ===================================================================


    ######################
    # From the RoboCompCamera360RGB you can call this methods:
    # self.camera360rgb_proxy.getROI(...)

    ######################
    # From the RoboCompByteTrack you can use this types:
    # RoboCompByteTrack.Targets
    #
    # def ByteTrack_getTargets(self, ps, pb, clases):
    #     ret = ifaces.RoboCompByteTrack.OnlineTargets()
    #     scores = np.array(ps)
    #     boxes = np.array(pb)
    #     clases = np.array(clases)
    #     for i in self.tracker.update_original(scores, boxes, clases):
    #         target = ifaces.RoboCompByteTrack.Targets()
    #         tlwh = ifaces.RoboCompByteTrack.Box(i.tlwh)
    #         target.trackid = i.track_id
    #         target.score = i.score
    #         target.tlwh = tlwh
    #         target.clase = i.clase
    #         ret.append(target)
    #     return ret
    # #
    # # IMPLEMENTATION of getTargetswithdepth method from ByteTrack interface
    # #
    # def ByteTrack_getTargetswithdepth(self, ps, pb, depth, clases):
    #     ret = ifaces.RoboCompByteTrack.OnlineTargets()
    #     depth = np.frombuffer(depth.depth, dtype=np.float32).reshape(depth.height, depth.width, 1)
    #     scores = np.array(ps)
    #     boxes = np.array(pb)
    #     clases = np.array(clases)
    #     for i in self.tracker.update2(scores, boxes, depth, clases):
    #         target = ifaces.RoboCompByteTrack.Targets()
    #         tlwh = ifaces.RoboCompByteTrack.Box(i.tlwh)
    #         target.trackid = i.track_id
    #         target.score = i.score
    #         target.tlwh = tlwh
    #         target.clase = i.clase
    #         ret.append(target)
    #     return ret
    # # ===================================================================
    # # ===================================================================
    #
    # def ByteTrack_setTargets(self, ps, pb, clases, sender):
    #
    #
    # def ByteTrack_allTargets(self):
    #     return self.read_tracks

    # From the RoboCompLidar3D you can call this methods:
    # self.lidar3d_proxy.getLidarData(...)

    ######################
    # From the RoboCompLidar3D you can use this types:
    # RoboCompLidar3D.TPoint

    ######################
    # From the RoboCompByteTrack you can use this types:
    # RoboCompByteTrack.Targets