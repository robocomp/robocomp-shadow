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
                    rgb = self.camera360rgb_proxy.getROI(-1, -1, -1, -1, -1, -1)

                    print("Camera specs:")
                    print(" width:", rgb.width)
                    print(" height:", rgb.height)
                    print(" depth", rgb.depth)
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
        try:
        	self.display = (params["display"] == "true" or params["display"] == "True")
        except:
        	traceback.print_exc()
        	print("Error reading config params")
        return True

    @QtCore.Slot()
    def compute(self):
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
            self.convert_to_visualelements_structure(rois)

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
                rgb = self.camera360rgb_proxy.getROI(self.roi_xcenter, self.roi_ycenter, self.roi_xsize,
                                                      self.roi_ysize, self.final_xsize, self.final_ysize)
                rgb_frame = np.frombuffer(rgb.image, dtype=np.uint8).reshape((rgb.height, rgb.width, 3))
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
        mask = np.zeros((seg.shape[0], seg.shape[1], 1), dtype=np.uint8)  # height, width, 3
        # mask[(seg == 3) | (seg == 91) | (seg == 52), :] = 255
        mask[(seg == 3), :] = 255
        return mask

    def extract_roi_instances_seg(self, instance_img, label):
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
        box = cv2.boundingRect(mask)
        left = int(box[0])
        right = int(box[0]+box[2])
        top = int(box[1])
        bot = int(box[1]+box[3])
        return ifaces.RoboCompVisualElements.TObject(id=id_, type=str(type_), left=left, top=top,
                                                     right=right, bot=bot, score=0.7,
                                                     depth=0, x=0, y=0, z=0)

    def show_fps(self, alive_time, period):
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
