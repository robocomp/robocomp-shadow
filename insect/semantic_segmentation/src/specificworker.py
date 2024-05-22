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

from PySide2.QtCore import QTimer
from PySide2.QtWidgets import QApplication
from rich.console import Console
from genericworker import *
import interfaces as ifaces
import traceback
from threading import Thread, Event
# Runs with pytorch 1.10.1
import torch
from PIL import Image
from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation
import time
import queue
import cv2
import numpy as np
import itertools

console = Console(highlight=False)


class SpecificWorker(GenericWorker):
    class TBox:
        def __init__(self, id_, type_, rect_, score_, depth_, x_, y_, z_):
            """
            Initializes a `BBox` instance with a set of input parameters: `id`,
            `type`, `roi`, `score`, `depth`, `x`, `y`, and `z`. These parameters
            determine the object's properties and values.

            Args:
                id_ (int): 0-based index of the bounding box in the list of boxes
                    belonging to the same object.
                type_ (str): type of the bounding box, which can be one of three
                    possible values: 'rect', 'polygon', or 'elliptical'.
                rect_ (tuple): 2D bounding rectangle of the object in the image,
                    which is used to compute the score and depth of the object.
                score_ (0-1 real number.): 0-1 probability value for the bounding
                    box being a positive example.
                    
                    		- `score_`: A 0-dimensional numpy array representing the
                    confidence score of the object detection. The values in the
                    array correspond to the objects present in the image and their
                    respective confidence scores.
                    		- `depth_`: A 1-dimensional numpy array representing the
                    distance from the center of the bounding box to the center of
                    the object in millimeters. This attribute helps determine the
                    size of the object in the image.
                depth_ (number/double value.): distance to the bounding box center
                    in millimeters.
                    
                    		- `depth_`: This attribute represents the distance to the
                    bbox center in millimeters (mm).
                    		- `x_`, `y_`, and `z_`: These attributes represent the
                    coordinates of the ROI center in camera coordinates.
                x_ (Coordinate (in a specific format).): 2D center coordinate of
                    the rectangle of interest (ROI) in camera coordinates.
                    
                    		- `x_`: The center coordinate of the bounding box (BX) in
                    pixels, represented as a tuple `(x, y)` containing the coordinates
                    of the BX center.
                y_ (double.): 2D coordinates of the rectangle's center in camera
                    space.
                    
                    		- `x_`: The center coordinates of the ROI in the image,
                    represented as (x, y) values in camera coordinates.
                    		- `y_`: The center coordinates of the ROI in the image,
                    represented as (x, y) values in camera coordinates. This
                    property may require destruction if it is a deserialized input
                    and needs to be processed further.
                    		- `z_`: The distance from the bbox center to the ROI center
                    in mm.
                    		- `score_`: A floating-point value representing the score
                    of the bbox, where 0 indicates no score and 1 indicates the
                    highest possible score.
                    		- `depth_`: The distance to the bbox center in mm.
                z_ (float): 3D position of the ROI center in the world coordinate
                    system.

            """
            self.id = id_
            self.type = type_
            self.roi = rect_
            self.score = score_  # 0 - 1
            self.depth = depth_  # distance to bbox center in mm
            self.x = x_  # roi center coordinates in camera
            self.y = y_
            self.z = z_
            self.display = False

    def __init__(self, proxy_map, startup_check=False):
        """
        Initializes various elements such as image queues, a mouse callback for
        the window with the segmentation results, camera parameters, and a timer
        for the compute function. It also creates a thread for reading images from
        the camera.

        Args:
            proxy_map (ndarray (or NumPy array).): 3D object map that maps from
                points in the real world to points in the virtual world of the
                360-degree camera, allowing the sematic segmentation to correctly
                interpret the positions and orientations of objects in the virtual
                world based on their actual locations in the real world.
                
                	1/ `proxy`: This is a dictionary that maps the label index (starting
                from 0) to the corresponding Proxy API endpoint name. The keys in
                this dictionary are the labels that can be recognized by the model,
                and the values are the names of the Proxy APIs that handle each
                label. For example, the key 'door' in `proxy_map` maps to the Proxy
                API 'doorgui'.
                	2/ `label_map`: This is a dictionary that maps the index of each
                label in the `proxy_map` to its corresponding semantic segmentation
                label (starting from 1). The keys in this dictionary are the label
                indices, and the values are the label names. For example, the key
                'door' in `label_map` maps to the label 'building'.
                	3/ `crs`: This is a tuple containing the intrinsic camera matrix
                (rotation and translation vectors) for the current camera view.
                	4/ `image_dims`: This is a tuple containing the width and height
                of the input image in pixels.
                	5/ `image_height`: This is the height of the input image in pixels.
                	6/ `event`: This is an event that triggered the creation of the
                Semantic Segmentor, which can be any of the events supported by
                the PyTorch framework (such as a keyboard key press or a mouse
                button click).
                
                	The `read_queue` and `camera_name` variables are also explained:
                
                		- `read_queue`: This is a queue that stores the input images to
                be processed by the Semantic Segmentor. The Semantic Segmentor
                reads images from this queue using a separate thread.
                		- `camera_name`: This is the name of the camera view that the
                Semantic Segmentor is currently processing.
                
                	The function also sets up event handling and a timer to call the
                `compute` method periodically.
            startup_check (int): 1st time the Semantic Segmentator has been run,
                and it is used to initialize the image queue for the read thread
                when it is set to `True`.

        """
        super(SpecificWorker, self).__init__(proxy_map)
        self.Period = 100
        self.thread_period = 100
        if startup_check:
            self.startup_check()
        else:
            # keyboard event to stop thread
            self.event = Event()

            self.objects_read = []
            self.objects_write = []

            # Hz
            self.cont = 0
            self.last_time = time.time()
            self.fps = 0

            # load Mask2Former fine-tuned on ADE20k semantic segmentation
            self.processor = AutoImageProcessor.from_pretrained("facebook/mask2former-swin-tiny-ade-semantic")
            self.model = Mask2FormerForUniversalSegmentation.from_pretrained(
                "facebook/mask2former-swin-tiny-ade-semantic")

            # self.model = SamModel.from_pretrained("facebook/sam-vit-base")
            # self.processor = SamProcessor.from_pretrained("facebook/sam-vit-base")

            # self.processor = AutoImageProcessor.from_pretrained("facebook/mask2former-swin-large-ade-panoptic")
            # self.model = Mask2FormerForUniversalSegmentation.from_pretrained(
            #     "facebook/mask2former-swin-large-ade-panoptic")
            self.color_palette = [list(np.random.choice(range(256), size=3)) for _ in
                                  range(len(self.model.config.id2label))]
            self.model = self.model.to('cuda:0')
            self.labels = {
                'wall': 0,
                'building': 1,
                'sky': 2,
                'floor': 3,
                'tree': 4,
                'ceiling': 5,
                'road': 6,
                'bed ': 7,
                'windowpane': 8,
                'grass': 9,
                'cabinet': 10,
                'sidewalk': 11,
                'person': 12,
                'earth': 13,
                'door': 14,
                'table': 15,
                'mountain': 16,
                'plant': 17,
                'curtain': 18,
                'chair': 19,
                'car': 20,
                'water': 21,
                'painting': 22,
                'sofa': 23,
                'shelf': 24,
                'house': 25,
                'sea': 26,
                'mirror': 27,
                'rug': 28,
                'field': 29,
                'armchair': 30,
                'seat': 31,
                'fence': 32,
                'desk': 33,
                'rock': 34,
                'wardrobe': 35,
                'lamp': 36,
                'bathtub': 37,
                'railing': 38,
                'cushion': 39,
                'base': 40,
                'box': 41,
                'column': 42,
                'signboard': 43,
                'chest of drawers': 44,
                'counter': 45,
                'sand': 46,
                'sink': 47,
                'skyscraper': 48,
                'fireplace': 49,
                'refrigerator': 50,
                'grandstand': 51,
                'path': 52,
                'stairs': 53,
                'runway': 54,
                'case': 55,
                'pool table': 56,
                'pillow': 57,
                'screen door': 58,
                'stairway': 59,
                'river': 60,
                'bridge': 61,
                'bookcase': 62,
                'blind': 63,
                'coffee table': 64,
                'toilet': 65,
                'flower': 66,
                'book': 67,
                'hill': 68,
                'bench': 69,
                'countertop': 70,
                'stove': 71,
                'palm': 72,
                'kitchen island': 73,
                'computer': 74,
                'swivel chair': 75,
                'boat': 76,
                'bar': 77,
                'arcade machine': 78,
                'hovel': 79,
                'bus': 80,
                'towel': 81,
                'light': 82,
                'truck': 83,
                'tower': 84,
                'chandelier': 85,
                'awning': 86,
                'streetlight': 87,
                'booth': 88,
                'television receiver': 89,
                'airplane': 90,
                'dirt track': 91,
                'apparel': 92,
                'pole': 93,
                'land': 94,
                'bannister': 95,
                'escalator': 96,
                'ottoman': 97,
                'bottle': 98,
                'buffet': 99,
                'poster': 100,
                'stage': 101,
                'van': 102,
                'ship': 103,
                'fountain': 104,
                'conveyer belt': 105,
                'canopy': 106,
                'washer': 107,
                'plaything': 108,
                'swimming pool': 109,
                'stool': 110,
                'barrel': 111,
                'basket': 112,
                'waterfall': 113,
                'tent': 114,
                'bag': 115,
                'minibike': 116,
                'cradle': 117,
                'oven': 118,
                'ball': 119,
                'food': 120,
                'step': 121,
                'tank': 122,
                'trade name': 123,
                'microwave': 124,
                'pot': 125,
                'animal': 126,
                'bicycle': 127,
                'lake': 128,
                'dishwasher': 129,
                'screen': 130,
                'blanket': 131,
                'sculpture': 132,
                'hood': 133,
                'sconce': 134,
                'vase': 135,
                'traffic light': 136,
                'tray': 137,
                'ashcan': 138,
                'fan': 139,
                'pier': 140,
                'crt screen': 141,
                'plate': 142,
                'monitor': 143,
                'bulletin board': 144,
                'shower': 145,
                'radiator': 146,
                'glass': 147,
                'clock': 148,
                'flag': 149
            }
            self.selected_labels = {
                'bed ': 7,
                'windowpane': 8,
                'cabinet': 10,
                'person': 12,
                'earth': 13,
                'door': 14,
                'table': 15,
                'plant': 17,
                'curtain': 18,
                'chair': 19,
                'painting': 22,
                'sofa': 23,
                'shelf': 24,
                'mirror': 27,
                'armchair': 30,
                'seat': 31,
                'desk': 33,
                'wardrobe': 35,
                'lamp': 36,
                'bathtub': 37,
                'railing': 38,
                'cushion': 39,
                'base': 40,
                'box': 41,
                'column': 42,
                'signboard': 43,
                'chest of drawers': 44,
                'counter': 45,
                'sink': 47,
                'refrigerator': 50,
                'grandstand': 51,
                'stairs': 53,
                'runway': 54,
                'case': 55,
                'pillow': 57,
                'stairway': 59,
                'bookcase': 62,
                'blind': 63,
                'coffee table': 64,
                'toilet': 65,
                'flower': 66,
                'book': 67,
                'bench': 69,
                'countertop': 70,
                'stove': 71,
                'palm': 72,
                'kitchen island': 73,
                'computer': 74,
                'swivel chair': 75,
                'bar': 77,
                'arcade machine': 78,
                'light': 82,
                'television receiver': 89,
                'bottle': 98,
                'buffet': 99,
                'poster': 100,
                'stage': 101,
                'basket': 112,
                'bag': 115,
                'oven': 118,
                'ball': 119,
                'food': 120,
                'step': 121,
                'tank': 122,
                'microwave': 124,
                'pot': 125,
                'dishwasher': 129,
                'screen': 130,
                'vase': 135,
                'tray': 137,
                'ashcan': 138,
                'plate': 142,
                'monitor': 143,
                'radiator': 146,
                'glass': 147,
                'clock': 148,
            }

            self.camera_name = "/Shadow/camera_top"
            self.winname = "Semantic Segmentator"
            cv2.namedWindow(self.winname)
            cv2.setMouseCallback(self.winname, self.mouse_click)

            # queue for image reading thread
            self.read_queue = queue.Queue(1)

            self.segmented_img = None
            self.instance_img = None
            self.mask_image = None
            self.id_to_label = {}

            # read test image to get sizes
            started_camera = False
            while not started_camera:
                try:
                    rgb = self.camera360rgbd_proxy.getROI(-1, -1, -1, -1, -1, -1)

                    print("Camera specs:")
                    print(" width:", rgb.width)
                    print(" height:", rgb.height)
                    print(" focalx", rgb.focalx)
                    print(" focaly", rgb.focaly)
                    print(" period", rgb.period)
                    print(" ratio {:.2f}.format(image.width/image.height)")

                    # Image ROI require parameters
                    self.final_xsize = 384
                    self.final_ysize = 384
                    self.roi_xsize = rgb.width // 2
                    self.roi_ysize = rgb.height
                    self.roi_xcenter = rgb.width // 2
                    self.roi_ycenter = rgb.height // 2

                    started_camera = True
                except Ice.Exception as e:
                    traceback.print_exc()
                    print(e, "Trying again...")
                    time.sleep(2)

            # Thread to read images
            self.read_thread = Thread(target=self.get_rgb_thread, args=[self.event],
                                      name="read_queue", daemon=True)
            self.read_thread.start()

            # init compute
            self.timer.timeout.connect(self.compute)
            self.timer.start(self.Period)

    def __del__(self):
        """Destructor"""

    def setParams(self, params):
        """
        Sets a parameter named 'display' to either true or false depending on its
        value in the input dictionary passed to it.

        Args:
            params (dict): configuration parameters passed to the function, which
                are then accessed using square bracket notation (`[]`) to determine
                whether the `display` variable is set to `true` or not.

        Returns:
            int: a boolean value indicating whether the display parameter is set
            to true or false.

        """
        try:
        	self.display = (params["display"] == "true" or params["display"] == "True")
        except:
        	traceback.print_exc()
        	print("Error reading config params")
        return True

    @QtCore.Slot()
    def compute(self):
        """
        Performs post-processing on semantic segmentation outputs from a processor,
        creating instance masks and ROIs for selected classes (door). It then
        visualizes the segmentation results and displays FPS information.

        """
        now = time.time()
        try:
            rgb_frame, outputs, alive_time, im_pil_size, period = self.read_queue.get()

            # postprocess segmentation result
            self.segmented_img = self.processor.post_process_semantic_segmentation(outputs, target_sizes=[im_pil_size])[
                0].cpu()
            self.instance_img = self.processor.post_process_instance_segmentation(outputs)[0]
            print("TIME EXPENDED 1", time.time() - now)

            # create masks from segmented img
            self.mask_image = self.create_mask(self.segmented_img)
            print("TIME EXPENDED 2", time.time() - now)
            # extract rois from selected classes (door)
            rois, masks = self.extract_roi_instances_seg(self.instance_img, 14)
            print("TIME EXPENDED 3", time.time() - now)
            # create Ice interface data structure and send to ByteTracker
            # self.convert_to_visualelements_structure(rois)

            if self.display:
                frame = self.draw_semantic_segmentation(self.winname, rgb_frame, self.segmented_img, rois)
                cv2.imshow(self.winname, frame)
                cv2.waitKey(2)

        except:
            print("Error communicating with CameraRGBDSimple")
            traceback.print_exc()

        # FPS
        try:
            self.show_fps(alive_time, period)
        except KeyboardInterrupt:
            self.event.set()
         #print("Elapsed:", int((time.time() - now)*1000), " msecs")

    ##################################################################################################
    def get_rgb_thread(self, event: Event):
       ''' 
       :param event: keyboard event to stop the thread
       :return:
       '''

       while not event.is_set():
            now = time.time()
            try:
                rgb = self.camera360rgbd_proxy.getROI(self.roi_xcenter, self.roi_ycenter, self.roi_xsize,
                                                      self.roi_ysize, self.final_xsize, self.final_ysize)
                rgb_frame = np.frombuffer(rgb.rgb, dtype=np.uint8).reshape((rgb.height, rgb.width, 3))
                img = cv2.cvtColor(rgb_frame, cv2.COLOR_BGR2RGB)
                im_pil = Image.fromarray(img)
                inputs = self.processor(images=im_pil, return_tensors="pt").to('cuda:0')
                delta = int(1000 * time.time() - rgb.alivetime)
                with torch.no_grad():
                    outputs = self.model(**inputs)
                self.read_queue.put([img, outputs, delta, im_pil.size[::-1], rgb.period])
                event.wait(self.thread_period/1000)

            except Ice.Exception as e:
                print(e, "Error communicating with CameraRGBDSimple")
                traceback.print_exc()

    def create_mask(self, seg):
        """
        Creates a binary mask from an input segmatary array based on three specific
        segment values. The mask elements corresponding to these values are set
        to 255, while all other elements remain 0.

        Args:
            seg (ndarray object, more particularly an image.): 2D image of the
                objects that need to be masked.
                
                		- `shape[0]`: The number of pixels in the horizontal direction.
                		- `shape[1]`: The number of pixels in the vertical direction.
                		- `dtype`: The data type of the pixel values, which is inferred
                to be `np.uint8`.
                		- `seg`: The deserialized input array with shape `(n_pixels,)`
                containing the segmentation mask values.
                		- `(seg == 3)`: A boolean vector indicating the locations where
                the segmentation value is equal to 3.
                		- `(seg == 91)`: A boolean vector indicating the locations where
                the segmentation value is equal to 91.
                		- `(seg == 52)`: A boolean vector indicating the locations where
                the segmentation value is equal to 52.

        Returns:
            int: a binary mask where pixels with values of 3, 91, or 52 are set
            to 255.

        """
        mask = np.zeros((seg.shape[0], seg.shape[1], 1), dtype=np.uint8)  # height, width, 3
        # mask[(seg == 3) | (seg == 91) | (seg == 52), :] = 255
        mask[(seg == 3), :] = 255
        return mask

    def extract_roi_instances_seg(self, instance_img, label):
        """
        Generates a list of bounding boxes and their corresponding masks for each
        instance of a given label in an image. It does this by iterating through
        instances in the image, applying a threshold to determine if they belong
        to the desired label, and then generating a box and mask for each instance
        that passes the threshold.

        Args:
            instance_img (3D NumPy array.): 2D instance segmentation image that
                contains the objects to be segmented.
                
                		- `labels`: a dictionary of label IDs and their corresponding classes.
                		- `segments_info`: a list of dictionaries containing information
                about each instance in the image, including the ID, class label,
                and score (a value between 0 and 1 indicating the instance's
                confidence level).
                		- `instance_img`: an array containing the image data with labeled
                instances.
                
                	The function iterates over each instance in the image and extracts
                the corresponding ROIs using the `do_box` function. The returned
                boxes are added to a list, and the masks associated with each
                instance are added to another list.
            label (int): 3D label that is being detected and segmented in the
                instance images.

        Returns:
            list: a list of bounding boxes and their corresponding segmentation
            masks for the specified label in the input image.

        """
        inst = instance_img['segments_info']
        ids_list = []
        # for sl_key, sl_val in self.labels.items():
        #     ids_list.append([[v['id'] for v in inst if v['label_id'] == label and v['score'] > 0.7], sl_key])
        for v in inst:
            if v['label_id'] == label:
                ids_list.append(v['id'])

        inst_img = instance_img['segmentation']

        box_list = []
        mask_list = []
        # for ids, cls in ids_list:
        for id_ in ids_list:
            mask = np.zeros((inst_img.shape[0], inst_img.shape[1], 1), dtype=np.uint8)
            mask[inst_img == id_] = 255
            tbox = self.do_box(id_, label, mask, inst_img.shape)
            box_list.append(tbox)
            mask_list.append(mask)
        return box_list, mask_list

    def convert_to_visualelements_structure(self, rois):
        """
        Converts a list of ROIs to a list of `TObjects` instances representing
        visual elements, and updates the internal `objects_write` and `objects_read`
        lists.

        Args:
            rois (`ifaces.RoboCompVisualElements.TRoi` object(s).): 2D ROIs (regions
                of interest) defined by the user and provides them to the function
                for generating visual objects based on their type, size, location,
                and score.
                
                		- `type`: An integer value indicating the type of ROI (Region
                of Interest) element, which can be one of the following values:
                `0` for a rectangle, `1` for an ellipse, or `2` for a polyline.
                		- `left`, `top`, `right`, and `bot`: The coordinates of the
                upper-left corner of the ROI element in pixel coordinates.
                		- `score`: An integer value indicating the confidence score of
                the ROI element.
                		- `roi`: A `TRoi` object containing the dimensions and center
                coordinates of the ROI element, as well as its final size after
                image processing.

        """
        self.objects_write = ifaces.RoboCompVisualElements.TObjects()
        for roi in rois:
            act_object = ifaces.RoboCompVisualElements.TObject()
            act_object.type = int(roi.type)
            act_object.left = roi.left
            act_object.top = roi.top
            act_object.right = roi.right
            act_object.bot = roi.bot
            act_object.score = roi.score
            act_object.roi = ifaces.RoboCompVisualElements.TRoi(xcenter=self.roi_xcenter, ycenter=self.roi_ycenter,
                                                                xsize=self.roi_xsize, ysize=self.roi_ysize,
                                                                finalxsize=self.final_xsize, finalysize=self.final_ysize)
            self.objects_write.append(act_object)
        # self.objects_write = self.visualelements_proxy.getVisualObjects(self.objects_write)

        # swap
        self.objects_write, self.objects_read = self.objects_read, self.objects_write

        # try:
        #     self.visualelements_proxy.setVisualObjects(objects, 1)
        # except Ice.Exception as e:
        #     print(e, "Error communicating with ByteTracker")

    ############################## UTILS ###############################################

    def draw_panoptic_segmentation(self, winname, color_image, seg, rois):
        """
        Takes as input a panoptic segmentation mask, an image, and ROIs (regions
        of interest). It then applies coloring to the mask based on a provided
        palette, and displays the colored mask with bounding boxes overlaid on the
        image.

        Args:
            winname (str): window name for displaying the annotated image and its
                corresponding mask.
            color_image (int): 3D numpy array containing the original image data
                that will be processed and masked based on the ROI labels.
            seg (ndarray or NumPy array of shape `(n, m, 3)` where `n` and `m` are
                the number of channels and height/width pixels of an image,
                respectively.): 2D label image of the object of interest, which
                is used to determine the corresponding color value for each pixel
                in the image.
                
                	1/ `shape`: The shape of the input segmnetation tensor, which
                represents the number of rows, columns, and channels (3) in the
                input data.
                	2/ `dtype`: The data type of the input segmnetation tensor, which
                is assumed to be `np.uint8` in this example.
                	3/ `palette`: An array containing the color palette for the
                segmentation, which is used to convert the label values to
                corresponding colors.
                	4/ `seg`: The input segmentation tensor, which contains the label
                values for each pixel in the image.
            rois (int): 2D regions of interest (ROIs) in the image, which are used
                to display text annotations on the image with their corresponding
                type and ID.

        Returns:
            int: a binary mask and an image with segmented objects and their
            corresponding class labels.

        """
        color_seg = np.zeros((seg.shape[0], seg.shape[1], 3), dtype=np.uint8)
        palette = np.array(self.color_palette)
        for label, color in enumerate(palette):
            color_seg[seg == label, :] = color
        # Convert to BGR
        color_seg = color_seg[..., ::-1]

        # Show image + mask
        img = np.array(color_image) * 0.6 + color_seg * 0.4
        img = img.astype(np.uint8)
        txt_color = (0, 0, 255)
        font = cv2.FONT_HERSHEY_SIMPLEX

        if rois is not None:
            for r in rois:
                # print(r)
                text = '{}-{}-{:.2f}'.format(r.type, r.id, r.y)
                txt_size = cv2.getTextSize(text, font, 0.4, 1)[0]
                # print((r.left*self.x_ratio, r.top*self.y_ratio), (r.right*self.x_ratio, r.bot*self.y_ratio))
                # cv2.rectangle(img, (r.left*self.x_ratio, r.top*self.y_ratio), (r.right*self.x_ratio, r.bot*self.y_ratio), (0, 0, 255), 2)
                cv2.rectangle(img, (r.left, r.top), (r.right, r.bot), (0, 0, 255), 2)
                cv2.putText(img, text, (r.left*self.x_ratio, r.top*self.y_ratio + txt_size[1]), font, 0.4, txt_color, thickness=1)
        return img

    def draw_semantic_segmentation(self, winname, color_image, seg, rois):
        """
        Takes a semantic segmentation mask and a ROI (region of interest) array
        as input and generates an image with labeled regions and bounding boxes
        around objects of interest.

        Args:
            winname (int): name of the window to be displayed with the annotations.
            color_image (ndarray or numerical array.): 3D array of RGB values that
                make up the image to be labeled, which is used to calculate the
                background color and create the output image with the segmented objects.
                
                		- `color_image`: A 3D numpy array with shape `(H, W, 3)`
                representing the color image to be drawn on top of the semantic segmentation.
                		- `H` and `W`: The height and width of the color image, respectively.
                		- `3`: The number of color channels in the image (red, green,
                and blue).
            seg (int): 2D binary mask image of the object of interest, which is
                used to segment and highlight the object in the original RGB image.
            rois (ndarray (or NumPy array).): 2D regions of interest (ROIs) to be
                overlaid on the image and mask, which can include class labels or
                other information.
                
                		- `type`: A string indicating the type of region of interest
                (ROI) (e.g., "person", "car", etc.)
                		- `id`: An integer identifying the specific instance of the ROI
                within its type (e.g., 1, 2, etc.)
                		- `x_ratio`: A float representing the horizontal alignment of
                the ROI within the image
                		- `y_ratio`: A float representing the vertical alignment of the
                ROI within the image
                
                	These properties are used to draw bounding rectangles and text
                labels around each ROI in the input image, as well as to determine
                the corresponding class label for each ROI.

        Returns:
            int: an image with bounding boxes and text annotations for detected objects.

        """
        color_seg = np.zeros((seg.shape[0], seg.shape[1], 3), dtype=np.uint8)
        palette = np.array(self.color_palette)
        for label, color in enumerate(palette):
            color_seg[seg == label, :] = color

        # Convert to BGR
        # color_seg = color_seg[..., ::-1]

        # Show image + mask
        img = np.array(color_image) * 0.6 + color_seg * 0.4
        img = img.astype(np.uint8)
        txt_color = (0, 0, 255)
        font = cv2.FONT_HERSHEY_SIMPLEX

        if rois is not None:
            for r in rois:
                # print(r)
                text = '{}-{}-{:.2f}'.format(r.type, r.id, r.y)
                txt_size = cv2.getTextSize(text, font, 0.4, 1)[0]
                # cv2.rectangle(img, (r.left*self.x_ratio, r.top*self.y_ratio), (r.right*self.x_ratio, r.bot*self.y_ratio), (0, 0, 255), 2)
                cv2.rectangle(img, (r.left, r.top), (r.right, r.bot), (0, 0, 255), 2)
                cv2.putText(img, text, (r.left, r.top + txt_size[1]), font, 0.4, txt_color, thickness=1)
        return img

    # Given a class, return all instances in a class
    def extract_roi_instances_pan(self, img_dict, img_seg, label):
        """
        Generates high-quality documentation for code given to it. It takes a list
        of image segmentations, a dictionary of labels, and returns a list of
        bounding boxes and a list of masks corresponding to the labels in the segmentations.

        Args:
            img_dict (dict): 2D image data with labels, from which the function
                extracts information about the labels and their corresponding
                bounding boxes.
            img_seg (ndarray.): 2D image segmentation output of the network, which
                is used to compute the corresponding bounding box coordinates and
                mask values for each label ID in the given image.
                
                		- `shape`: The shape of the input image, which is (height, width,
                channels).
                		- "info": A dictionary containing information about the image
                segments, including their IDs and labels.
                		- "segments_info": An list of dictionaries, where each dictionary
                represents a segment in the image with its ID and label.
                
                	The function then destructures `img_seg` by extracting its ID and
                label attributes and using them to create boxes and masks for each
                segment in the image.
            label (int): 0-based integer index of the label to extract bounding
                boxes for from the provided image segmentation mask.

        Returns:
            tuple: a list of bounding box coordinates and their corresponding masks
            for the specified objects in an image.

        """
        ids = [x["id"] for x in img_dict["segments_info"] if x["label_id"] == label]
        boxes = []
        masks = []
        for id in ids:
            mask = np.zeros((img_seg.shape[0], img_seg.shape[1], 1), dtype=np.uint8)
            # mask[(seg == 3) | (seg == 91) | (seg == 52), :] = 255
            mask[(img_seg == id), :] = 255

            bbox = self.do_box(id, label, mask, img_seg.shape)
            boxes.append(bbox)
            masks.append(mask)
        return boxes, masks

    def do_box(self, id_, type_, mask, rgb_shape):
        """
        Computes a bounding box for an image mask using OpenCV and creates a
        `RoboCompVisualElements.TObject` instance with the computed bounding box
        coordinates, score, depth, and other metadata.

        Args:
            id_ (int): 4-byte value of an integer that uniquely identifies each
                instance of the `TObject` class being created, which is used to
                identify and reference the specific object within a visual scene.
            type_ (str): 3D shape type of the visual element being created.
            mask (`numpy.ndarray`.): 2D binary mask that is used to extract the
                rectangle coordinates of the object of interest in an image.
                
                		- `mask`: A NumPy array of shape (X, Y) containing binary pixel
                values representing a 2D mask for object detection.
            rgb_shape (ndarray object (e.g., array-like instance or a tensor).):
                2D shape of the object in the image as a bounding rectangle,
                providing its left and right coordinates and top and bottom coordinates.
                
                		- `id`: an integer identifier for the object, represented as `id_`.
                		- `type`: a string indicating the type of the object, represented
                as `str(type_)_.`
                		- `left`, `top`, `right`, and `bot`: integers representing the
                bounding box coordinates of the object.
                		- `score`: a real number representing the score of the object,
                set to 0.7 by default.
                		- `depth`: an integer representing the depth of the object, set
                to 0 by default.
                		- `x`, `y`, and `z`: integers representing the position of the
                object in the 3D space, set to 0 by default.

        Returns:
            instance of `TObject`, with fields representing various coordinates
            and attributes such as `id`, `type`, `left`, `top`, `right`, `bot`,
            `score`, `depth`, `x`, and `y`, all initialized to specific values:
            an instance of the `TObject` class from the RoboCompVisualElements package.
            
            		- `id`: The unique identifier for each object.
            		- `type`: The type of object (either `cv2.BOX_CYLINDROID` or `cv2.BOX_SPHERE`).
            		- `left`: The left coordinate of the bounding box.
            		- `top`: The top coordinate of the bounding box.
            		- `right`: The right coordinate of the bounding box.
            		- `bot`: The bottom coordinate of the bounding box.
            		- `score`: The score value (which is set to 0.7 by default).
            		- `depth`: The depth value, which is set to 0 by default.
            		- `x`: The x-coordinate of the bounding box.
            		- `y`: The y-coordinate of the bounding box.
            		- `z`: The z-coordinate of the bounding box.

        """
        box = cv2.boundingRect(mask)
        left = int(box[0])
        right = int(box[0]+box[2])
        top = int(box[1])
        bot = int(box[1]+box[3])
        return ifaces.RoboCompVisualElements.TObject(id=id_, type=str(type_), left=left, top=top,
                                                     right=right, bot=bot, score=0.7,
                                                     depth=0, x=0, y=0, z=0)

    def show_fps(self, alive_time, period):
        """
        Updates the display rate (fps) and the thread period based on a sliding
        window of 1 seconds (1000 milliseconds). It checks if a second has passed
        since the last update, and if so, calculates and prints the current fps,
        alive time, and period. It also updates the thread period using a clipped
        value between 0 and 200 milliseconds.

        Args:
            alive_time ('ms'.): total time that the current thread has been active,
                and it is used to calculate the current period and increment of
                the image processing cycle.
                
                		- `alive_time`: A float value representing the total time the
                program has been running in milliseconds (ms). It is updated at
                each call to `show_fps` with the current time delay between checks.
                		- `cont`: An integer value representing the number of intervals
                between successive `show_fps` calls, expressed in Hz (i.e., 1/cont
                represents the average rate at which the program is alive). It is
                used to calculate the interval between subsequent `show_fps` outputs.
                		- `period`: An integer value representing the duration of each
                interval in milliseconds (ms) from when the previous `show_fps`
                call was made until the next check is due.
                		- `delta`: A variable representing the increment of the wait
                time for the `show_fps` function's internal clock. Its value ranges
                from -1 to 1, and it is used to calculate the current wait time
                before incrementing the thread period.
            period (int): time interval between image acquisitions or other events
                and is used to calculate the alive time of the system.

        """
        if time.time() - self.last_time > 1:
            self.last_time = time.time()
            cur_period = int(1000./self.cont)
            delta = (-1 if (period - cur_period) < -1 else (1 if (period - cur_period) > 1 else 0))
            print("Freq:", self.cont, "Hz. Alive_time:", alive_time, "ms. Img period:", int(period),
                  "ms. Curr period:", cur_period, "ms. Increment:", delta, "Current wait time:", self.thread_period)
            self.thread_period = np.clip(self.thread_period+delta, 0, 200)
            self.cont = 0
        else:
            self.cont += 1

    def mouse_click(self, event, x, y, flags, param):
        """
        Updates the selected object and prints the current class label of the
        clicked point based on YOLO annotations. It then checks if the click was
        inside a YOLO object and sets the new target object if so.

        Args:
            event (`cv2.EventType` value, which belongs to a built-in type provided
                by OpenCV named 'cv2.EventType'.): MouseButtonEvent generated by
                OpenCV, providing the coordinates of the button press and the
                object class label associated with it.
                
                		- `cv2.EVENT_LBUTTONDOWN`: This event is triggered when the left
                button of the mouse is pressed down.
                		- `x`: The x-coordinate of the mouse click in the image coordinates.
                		- `y`: The y-coordinate of the mouse click in the image coordinates.
            x (float): 2D coordinates of the event, which can be used to determine
                the position of the selected object in the image.
            y (int): 2D coordinate of the point where the left mouse button was clicked.
            flags (float): event flag that indicates whether the mouse button was
                pressed on an object detected by YOLO, and it is used to determine
                which object to select or update the target object when a mouse
                button is pressed within a specific region of the image.
            param (float): 2D mouse position, which is used to determine which
                object in the list of detected objects (represented by the
                `self.labels` dictionary) was clicked and to set the new target
                object using the `self.yolo_objects` list.

        """
        if event == cv2.EVENT_LBUTTONDOWN:
            self.selected_object = None
            point = (x, y)
            print("Selected class: ", list(self.labels.keys())[list(self.labels.values()).index(self.segmented_img[y, x].item())])

            # check if clicked point on yolo object. If so, set it as the new target object
            # for b in self.yolo_objects:
            #     if x >= b.left and x < b.right and y >= b.top and y < b.bot:
            #         self.selected_object = b
            #         print("Selected yolo object", self.yolo_object_names[self.selected_object.type], self.selected_object==True)
            #         self.previous_yolo_id = None
            #         break

    ##############################################################################################3
    def startup_check(self):
        """
        Tests several interfaces from the `ifaces` module, including
        `RoboCompCameraRGBDSimple`, `RoboCompSemanticSegmentation`, and others.
        It runs unit tests on each interface using various methods to verify their
        functionality.

        """
        print(f"Testing RoboCompCameraRGBDSimple.Point3D from ifaces.RoboCompCameraRGBDSimple")
        test = ifaces.RoboCompCameraRGBDSimple.Point3D()
        print(f"Testing RoboCompCameraRGBDSimple.TPoints from ifaces.RoboCompCameraRGBDSimple")
        test = ifaces.RoboCompCameraRGBDSimple.TPoints()
        print(f"Testing RoboCompCameraRGBDSimple.TImage from ifaces.RoboCompCameraRGBDSimple")
        test = ifaces.RoboCompCameraRGBDSimple.TImage()
        print(f"Testing RoboCompCameraRGBDSimple.TDepth from ifaces.RoboCompCameraRGBDSimple")
        test = ifaces.RoboCompCameraRGBDSimple.TDepth()
        print(f"Testing RoboCompCameraRGBDSimple.TRGBD from ifaces.RoboCompCameraRGBDSimple")
        test = ifaces.RoboCompCameraRGBDSimple.TRGBD()
        print(f"Testing RoboCompSemanticSegmentation.TBox from ifaces.RoboCompSemanticSegmentation")
        test = ifaces.RoboCompSemanticSegmentation.TBox()
        QTimer.singleShot(200, QApplication.instance().quit)

    # =============== Ice service methods ===============================
    # ===================================================================##

    # IMPLEMENTATION of getNamesofCategories method from MaskElements interface
    #

    def MaskElements_getNamesofCategories(self):
        return self.labels

    #
    # IMPLEMENTATION of setMasks method from MaskElements interface
    #
    def MaskElements_getMasks(self, masks):
        """
        Creates a list of `ifaces.RoboCompMaskElements` objects representing the
        provided image mask. Each element in the list has the same dimensions as
        the input image and contains a copy of the image data. Additionally, the
        ROI (Region Of Interest) of each element is defined based on the input
        `roi_xcenter`, `roi_ycenter`, `roi_xsize`, and `roi_ysize` parameters.

        Args:
            masks (`ifaces.RoboCompMaskElements.TMask`.): 2D array of mask elements,
                which is converted into a list of `ifaces.RoboCompMaskElements`
                objects and used to define the ROI for each mask element in the list.
                
                		- `mask`: A `TMask` object with shape `(height, width)` and an
                image property of type `numpy.ndarray` of size `(height, width, 3)`.
                		- `roi`: An instance of `TRoi` with the following properties:
                		+ `xcenter`: An integer value representing the x-coordinate of
                the ROI's center.
                		+ `ycenter`: An integer value representing the y-coordinate of
                the ROI's center.
                		+ `xsize`: An integer value representing the width of the ROI.
                		+ `ysize`: An integer value representing the height of the ROI.
                		+ `finalxsize`: An integer value representing the desired output
                size of the ROI after filtering.
                		+ `finalysize`: An integer value representing the desired output
                size of the ROI after filtering.

        Returns:
            list: a list of `ifaces.RoboCompMaskElements` objects, each containing
            a mask image and a ROI (Region of Interest) specification.

        """
        mask_list = []
        mask = ifaces.RoboCompMaskElements.TMask()
        mask.width = self.mask_image.shape[1]
        mask.height = self.mask_image.shape[0]
        mask.image = self.mask_image.tobytes()
        mask.roi = ifaces.RoboCompMaskElements.TRoi(xcenter=self.roi_xcenter, ycenter=self.roi_ycenter,
                                                            xsize=self.roi_xsize, ysize=self.roi_ysize,
                                                            finalxsize=self.final_xsize, finalysize=self.final_ysize)
        mask_list.append(mask)
        return mask_list

    #
    # IMPLEMENTATION of getVisualObjects method from VisualElements interface
    #

    def VisualElements_getVisualObjects(self, objects):
        return self.objects_read

    # ===================================================================
    # ===================================================================
    ######################
    # From the RoboCompByteTrack you can call this methods:
    # self.bytetrack_proxy.allTargets(...)
    # self.bytetrack_proxy.getTargets(...)
    # self.bytetrack_proxy.getTargetswithdepth(...)
    # self.bytetrack_proxy.setTargets(...)

    ######################
    # From the RoboCompByteTrack you can use this types:
    # RoboCompByteTrack.Targets

    ######################
    # From the RoboCompCamera360RGB you can call this methods:
    # self.camera360rgb_proxy.getROI(...)

    ######################
    # From the RoboCompMaskElements you can use this types:
    # RoboCompMaskElements.TRoi
    # RoboCompMaskElements.TMask

    ######################
    # From the RoboCompVisualElements you can use this types:
    # RoboCompVisualElements.TRoi
    # RoboCompVisualElements.TObject

# @QtCore.Slot()
    # def compute(self):
    #     now = time.time()
    #     try:
    #         rgb_frame, outputs, alive_time, im_pil_size, period = self.read_queue.get()
    #         # you can pass them to processor for postprocessing
    #         self.panoptic_img = self.processor.post_process_panoptic_segmentation(outputs, target_sizes=[im_pil_size], label_ids_to_fuse=[3])[
    #             0]
    #         self.segmented_img = self.panoptic_img["segmentation"].cpu()
    #         # self.segmented_img = self.processor.post_process_semantic_segmentation(outputs, target_sizes=[im_pil_size])[
    #         #     0].cpu()
    #         rois = self.get_rois_from_labels(self.panoptic_img, self.segmented_img, 14)
    #         frame = self.draw_semantic_segmentation(self.winname, rgb_frame, self.segmented_img, rois)
    #         # cv2.imshow(self.winname, cv2.resize(frame, (600, 600)))
    #         cv2.imshow(self.winname, frame)
    #         cv2.waitKey(2)
    #
    #     except:
    #         print("Error communicating with CameraRGBDSimple")
    #         traceback.print_exc()
    #
    #     # FPS
    #     try:
    #         self.show_fps(alive_time, period)
    #     except KeyboardInterrupt:
    #         self.event.set()
    #     # print("Elapsed:", int((time.time() - now)*1000), " msecs")

 # #
 #    # IMPLEMENTATION of getInstances method from SemanticSegmentation interface
 #    #
 #    def SemanticSegmentation_getInstances(self):
 #        return self.rois
 #    #
 #    # IMPLEMENTATION of getInstancesImage method from SemanticSegmentation interface
 #    #
 #    def SemanticSegmentation_getInstancesImage(self):
 #        img = ifaces.RoboCompCameraSimple.TImage()
 #        img.image = self.instance_img
 #        img.height, img.width = self.instance_img.shape
 #        img.depth = 1
 #        return img
 #
 #    def SemanticSegmentation_getSegmentedImage(self):
 #        if self.segmented_img is not None:
 #            img = ifaces.RoboCompCameraSimple.TImage()
 #            img.image = bytes(list(itertools.chain(*self.segmented_img.tolist())))
 #            img.height, img.width = self.segmented_img.shape
 #            img.depth = 1
 #            return img
 #        else:
 #            print("Segmented image is None")
 #            return ifaces.RoboCompCameraSimple.TImage()
 #
 #    def SemanticSegmentation_getMaskedImage(self, category):
 #        if self.mask_image is not None:
 #            print("request")
 #            img = ifaces.RoboCompCameraSimple.TImage()
 #            img.image = self.mask_image
 #            img.height, img.width, img.depth = self.mask_image.shape
 #            return img
 #        else:
 #            return ifaces.RoboCompCameraSimple.TImage()
 #    # ===================================================================
