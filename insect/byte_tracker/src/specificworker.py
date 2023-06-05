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
        self.Period = 100

        # ROI parameters. Must be filled up
        self.final_xsize = 0
        self.final_ysize = 0
        self.roi_xcenter = 512
        self.roi_ycenter = 256
        self.roi_xsize = 512
        self.roi_ysize = 256
        self.original_fov = 360
        self.cam_to_lidar = self.make_matrix_rt(0, 0, 0, 0, 0,
                                                108.51)
        self.lidar_to_cam = np.linalg.inv(self.cam_to_lidar)
        self.lidar_to_cams = {"cam_front": self.make_matrix_rt(0, 0, 0, 0, 0, -108.51),
                              "cam_right": self.make_matrix_rt(0, 0, np.pi / 2, 0, 0, -108.51),
                              "cam_back_1": self.make_matrix_rt( 0, 0,  np.pi, 0, 0, -108.51),
                              "cam_back_2": self.make_matrix_rt( 0, 0,  np.pi, 0, 0, -108.51),
                              "cam_left": self.make_matrix_rt( 0, 0, -np.pi / 2, 0, 0, -108.51)}
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

        # ROI offsets respect the original image
        self.x_roi_offset = 0
        self.y_roi_offset = 0

        if startup_check:
            self.startup_check()
        else:
            # @dataclass
            # class TRoi:
            #     final_xsize: int = 0
            #     final_ysize: int = 0
            #     xcenter: int = 0
            #     ycenter: int = 0
            #     xsize: int = 0
            #     ysize: int = 0
            # self.roi = TRoi()

            self.objects_read = []
            self.objects_write = []
            self.display = False

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
            if params["depth_flag"] == "true" or params["depth_flag"] == "True":
                self.tracker = BYTETrackerDepth(frame_rate=30)
            else:
                self.tracker = BYTETracker(frame_rate=30)
            if params["display"] == "true" or params["display"] == "True":
                self.display = True
        except:
            traceback.print_exc()
            print("Error reading config params")

        return True


    @QtCore.Slot()
    def compute(self):
        t1 = time.time()
        # project lidar points on image
        self.lidar_in_image = self.lidar_points()
        # Get tracks from Bytetrack and convert data to VisualElements interface
        if self.display:
            img = self.display_data(self.read_image())
            if img is not None:
                cv2.imshow("Image", img)
                cv2.waitKey(1)
        self.show_fps()


    def read_visual_objects(self, visual_objects):
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
                bot=int(track.tlwh[1] + track.tlwh[3]), type=track.clase
            )
            for track in tracks
        ]
        self.objects_write = ifaces.RoboCompVisualElements.TObjects(self.distance_to_object(targets))

        # swap
        self.objects_write, self.objects_read = self.objects_read, self.objects_write
        return self.objects_read

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
        rgb = self.camera360rgb_proxy.getROI(-1, -1, -1, -1, -1, -1)

        # Reshape the 1D array of pixel data into a 3D array representing the RGB image.
        rgb_frame = np.frombuffer(rgb.image, dtype=np.uint8).reshape((rgb.height, rgb.width, 3))

        # Store the image dimensions for later use.
        self.width_img = rgb.width
        self.height_img = rgb.height


        return rgb_frame

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
            _ , element.depth = self.points_in_bbox(self.lidar_in_image, x0, y0, x1, y1)
        return objects


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
        # for i in self.lidar_in_image:
        #     cv2.circle(image, (int(i[0]), int(i[1])), 1, (0, 255, 0), 1)
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
            results, min_distance = self.points_in_bbox(self.lidar_in_image, x0, y0, x1, y1)
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
            # Put the text on the image.
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

    def lidar_points(self):
        """
        Extracts LIDAR points that lie within a specific region of interest (ROI)
        and projects them onto the image plane. Points are transformed according to
        their corresponding camera view, and those falling within image bounds are returned.

        Returns:
        lidar_in_image : numpy array
            An array of projected LIDAR points that fall within image bounds. Each point is
            represented as a 3D point [x, y, distance].

        """

        # Calculate angles for the region of interest.
        start_angle, len_angle = self.calculate_roi_angles()
        # Fetch LIDAR points that fall within the specified angle range.

        points = self.lidar3d_proxy.getLidarData(int(start_angle), int(len_angle))





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

        # cx_offsets = {"cam_front": self.width_img / 2,
        #               "cam_right": self.width_img / 2 + ((self.width_img / 2) / 2),
        #               "cam_left": self.width_img / 2 - ((self.width_img / 2) / 2),
        #               "cam_back_1": self.width_img, "cam_back_2": 0}
        # cy_offsets = {"cam_front": self.height_img / 2,
        #               "cam_right": self.height_img / 2 + ((self.height_img / 2) / 4),
        #               "cam_left": self.height_img / 2 + ((self.height_img / 2) / 4),
        #               "cam_back_1": self.height_img / 2,
        #               "cam_back_2": self.height_img / 2}

        # Define a small value to avoid division by zero.
        epsilon = 1e-7

        # Convert points into a numpy array and scale from millimeters to meters.
        points_array = np.array([[p.x, p.y, p.z] for p in points])

        # Add a fourth dimension to the points array for homogenous coordinates.
        points_array = np.append(points_array, np.ones((points_array.shape[0], 1)), axis=1)

        # Calculate the angle for each point.
        angles = np.arctan2(points_array[:, 0], points_array[:, 1])

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

                # Identify points that fall within the image bounds.
                valid_indices = np.where((0 <= x) & (x < 1024) & (0 <= y) & (y < 512))[0]

                # Add valid points to the lidar_in_image list.
                if len(valid_indices) > 0:
                    lidar_in_image.extend(np.column_stack([x[valid_indices], y[valid_indices],
                                                           np.linalg.norm(transformed_points[valid_indices, :3],
                                                                          axis=1)]).tolist())

        return np.array(lidar_in_image)

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

    # def make_matrix_rt(self, roll, pitch, heading, x0, y0, z0):
    #     a = roll
    #     b = pitch
    #     g = heading
    #     mat = np.array([[np.cos(b) * np.cos(g),
    #                      (np.sin(a) * np.sin(b) * np.cos(g) + np.cos(a) * np.sin(g)), (np.sin(a) * np.sin(g) -
    #                                                                                    np.cos(a) * np.sin(b) * np.cos(
    #                     g)), x0],
    #                     [-np.cos(b) * np.sin(g), (np.cos(a) * np.cos(g) - np.sin(a) * np.sin(b) * np.sin(g)),
    #                      (np.sin(a) * np.cos(g) + np.cos(a) * np.sin(b) * np.sin(g)), y0],
    #                     [np.sin(b), -np.sin(a) * np.cos(b), np.cos(a) * np.cos(b), z0],
    #                     [0, 0, 0, 1]])
    #     return mat

    def points_in_bbox(self, points, x1, y1, x2, y2):
        """
        Searches within a list of points for those located within a bounding box and finds the minimum
        distance among these points.

        Parameters:
        points : list
            A list of tuples, where each tuple represents a point and consists of (x, y, distance).
        x1, y1, x2, y2 : float
            The coordinates defining the bounding box. (x1, y1) represents the lower-left corner and
            (x2, y2) the upper-right corner.

        Returns:
        in_box_points : list
            A list of tuples, each tuple representing a point within the bounding box.
        min_distance : float
            The minimum distance among the points within the bounding box. If there are no points
            within the box, returns infinity.
        """
        in_box_points = [p for p in points if x1 <= p[0] <= x2 and y1 <= p[1] <= y2]
        min_distance = min((p[2] for p in in_box_points), default=float('inf'))
        return in_box_points, min_distance

    # def calculate_roi_angles(self):
    #     """
    #     This function calculates the start angle and the width in degrees of a region of interest (ROI) in an image.
    #     The image represents a 360-degree field of view (FOV).
    #
    #     The FOV is then transformed to a [0, 900] range, where 0 corresponds to a 180 degree angle in the original image,
    #     and 900 corresponds to a -180 degree angle.
    #
    #     The function assumes that the ROI is specified by its center (`self.roi_xcenter`) and its width (`self.roi_xsize`),
    #     and that the full width of the original image is stored in `self.width_img`.
    #
    #     Returns:
    #         start_angle: The starting angle of the ROI, transformed to the [0, 900] range.
    #         len_angle: The width of the ROI in degrees, in the transformed [0, 900] range.
    #     """
    #     # Original Field of View
    #     original_fov = 360
    #     # original_fov = 900
    #
    #     # Calculamos la proporción de la posición de inicio con respecto al ancho original
    #     start_ratio = self.width_img / 1024
    #
    #     # Calculamos el ángulo de inicio
    #     start_angle = original_fov * start_ratio - 180
    #     print(start_angle)
    #     # Compute the ratio and angles directly in one step
    #     start_angle = ((original_fov * (
    #                 self.roi_xcenter - self.roi_xsize // 2) / self.width_img - 180 - 180) / -360) * 900
    #     end_angle = ((original_fov * (
    #                 self.roi_xcenter + self.roi_xsize // 2) / self.width_img - 180 - 180) / -360) * 900
    #
    #     # Compute image width in degrees
    #     len_angle = abs(end_angle - start_angle)
    #
    #     print(start_angle, len_angle)
    #
    #     return start_angle, len_angle

    def calculate_roi_angles(self):
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

        data = self.read_visual_objects(objects)
        self.objects = objects
        # Get tracks from Bytetrack and convert data to VisualElements interface
        return self.to_visualelements_interface(self.tracker.update_original(np.array(data["scores"]),
                                                                                       np.array(data["boxes"]),
                                                                                       np.array(data["clases"]),
                                                                                       np.array(data["roi"])))

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