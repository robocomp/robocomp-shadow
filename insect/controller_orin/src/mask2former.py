''' Installation instructions


'''

## Runs with pytorch 1.10.1
import torch
import sys
from PIL import Image
from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation
import time
import cv2
import numpy as np

class Mask2Former():
    class TBox:
        def __init__(self, id_, type_, rect_, score_, depth_, x_, y_, z_):
            self.id = id_
            self.type = type_
            self.roi = rect_
            self.score = score_      # 0 - 1
            self.depth = depth_      # distance to bbox center in mm
            self.x = x_              # roi center coordinates in camera
            self.y = y_
            self.z = z_

    def __init__(self):
        # load Mask2Former fine-tuned on ADE20k semantic segmentation
        self.processor = AutoImageProcessor.from_pretrained("facebook/mask2former-swin-tiny-ade-semantic")
        self.model = Mask2FormerForUniversalSegmentation.from_pretrained("facebook/mask2former-swin-tiny-ade-semantic")
        self.color_palette = [list(np.random.choice(range(256), size=3)) for _ in range(len(self.model.config.id2label))]
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

    def process(self, img, depth_img, dfocalx, dfocaly ):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        im_pil = Image.fromarray(img)
        inputs = self.processor(images=im_pil, return_tensors="pt").to('cuda:0')

        with torch.no_grad():
            outputs = self.model(**inputs)

        # model predicts class_queries_logits of shape `(batch_size, num_queries)`
        # and masks_queries_logits of shape `(batch_size, num_queries, height, width)`
        #class_queries_logits = outputs.class_queries_logits
        #masks_queries_logits = outputs.masks_queries_logits

        # you can pass them to processor for postprocessing
        segmented_img = self.processor.post_process_semantic_segmentation(outputs, target_sizes=[im_pil.size[::-1]])[0].cpu()
        instance_img = self.processor.post_process_instance_segmentation(outputs)[0]
        mask = self.create_mask(segmented_img)
        rois = self.extract_roi_instances(instance_img, depth_img, dfocalx, dfocaly)
        return mask, segmented_img, rois

    def draw_semantic_segmentation(self, winname, color_image, seg):
        color_seg = np.zeros((seg.shape[0], seg.shape[1], 3), dtype=np.uint8)  # height, width, 3
        palette = np.array(self.color_palette)
        for label, color in enumerate(palette):
           color_seg[seg == label, :] = color
        # Convert to BGR
        color_seg = color_seg[..., ::-1]

        # Show image + mask
        img = np.array(color_image) * 0.5 + color_seg * 0.5
        img = img.astype(np.uint8)
        cv2.imshow(winname, np.asarray(img))
        cv2.waitKey(2)

    def create_mask(self, seg):
        mask = np.zeros((seg.shape[0], seg.shape[1], 1), dtype=np.uint8)  # height, width, 3
        #mask[(seg == 3) | (seg == 91) | (seg == 52), :] = 255
        mask[(seg == 3), :] = 255

        # labels_ids = torch.unique(seg).tolist()
        # for label_id in labels_ids:
        #     label = self.model.config.id2label[label_id]
        #     print(label, label_id)
        return mask


    def extract_roi_instances(self, instance_img, depth, dfocalx, dfocaly):
        inst = instance_img['segments_info']
        ids_list = []
        for sl_key, sl_val in self.selected_labels.items():
            ids_list.append([[v['id'] for v in inst if v['label_id'] == sl_val and v['score'] > 0.7], sl_key])

        inst_img = instance_img['segmentation']

        box_list = []
        for ids, cls in ids_list:
            for id_ in ids:
                mask = np.zeros((inst_img.shape[0], inst_img.shape[1], 1), dtype=np.uint8)
                mask[inst_img == id_] = 255
                tbox = self.do_box(id_, cls, mask, depth, inst_img.shape, dfocalx, dfocaly)
                box_list.append(tbox)
        return box_list

    def do_box(self, id_, type_, mask, depth, rgb_shape, dfocalx, dfocaly):
        box = cv2.boundingRect(mask)
        depth_to_rgb_factor_rows = rgb_shape[0] / depth.shape[0]
        depth_to_rgb_factor_cols = rgb_shape[1] / depth.shape[1]
        left = int(box[0] / depth_to_rgb_factor_cols)
        right = int(box[0]+box[2] / depth_to_rgb_factor_cols)
        top = int(box[1] / depth_to_rgb_factor_rows)
        bot = int(box[1]+box[3] / depth_to_rgb_factor_rows)
        roi = depth[top: top + bot, left: left + right]
        if roi.any():
            box_depth = np.min(roi) * 1000
        else:
            box_depth = -1
        y = box_depth
        cx_i = int(box[0] + box[2]/2)
        cy_i = int(box[1] + box[3]/2)
        cx = cx_i - depth.shape[1] / 2
        cy = cy_i - depth.shape[0] / 2
        x = cx * box_depth / dfocalx
        z = -cy * box_depth / dfocaly  # Z upwards

        return self.TBox(id_, type_, box, 0.7, box_depth, x, y, z)

    # box_list.append({"roi": cv2.boundingRect(mask),
    #                  "id": id_,
    #                  "type": cls,
    #                  })

      # for door_id in door_ids:
      #    mask = np.zeros((inst_img.shape[0], inst_img.shape[1], 1), dtype=np.uint8)
      #    mask[inst_img == door_id] = 255
      #    mask_23 = cv2.boundingRect(mask)
      #    #cv2.rectangle(frame, mask_23, (0, 0, 255), 2)


           # 'base': 40,
           # 'box': 41,
           # 'column': 42,
           # 'signboard': 43,
           # 'chest of drawers': 44,
           # 'counter': 45,
           # 'sand': 46,
           # 'sink': 47,
           # 'skyscraper': 48,
           # 'fireplace': 49,
           # 'refrigerator': 50,
           # 'grandstand': 51,
           # 'path': 52,
           # 'stairs': 53,
           # 'runway': 54,
           # 'case': 55,
           # 'pool table': 56,
           # 'pillow': 57,
           # 'screen door': 58,
           # 'stairway': 59,
           # 'river': 60,
           # 'bridge': 61,
           # 'bookcase': 62,
           # 'blind': 63,
           # 'coffee table': 64,
           # 'toilet': 65,
           # 'flower': 66,
           # 'book': 67,
           # 'hill': 68,
           # 'bench': 69,
           # 'countertop': 70,
           # 'stove': 71,
           # 'palm': 72,
           # 'kitchen island': 73,
           # 'computer': 74,
           # 'swivel chair': 75,
           # 'boat': 76,
           # 'bar': 77,
           # 'arcade machine': 78,
           # 'hovel': 79,
           # 'bus': 80,
           # 'towel': 81,
           # 'light': 82,
           # 'truck': 83,
           # 'tower': 84,
           # 'chandelier': 85,
           # 'awning': 86,
           # 'streetlight': 87,
           # 'booth': 88,
           # 'television receiver': 89,
           # 'airplane': 90,
           # 'dirt track': 91,
           # 'apparel': 92,
           # 'pole': 93,
           # 'land': 94,
           # 'bannister': 95,
           # 'escalator': 96,
           # 'ottoman': 97,
           # 'bottle': 98,
           # 'buffet': 99,
           # 'poster': 100,
           # 'stage': 101,
           # 'van': 102,
           # 'ship': 103,
           # 'fountain': 104,
           # 'conveyer belt': 105,
           # 'canopy': 106,
           # 'washer': 107,
           # 'plaything': 108,
           # 'swimming pool': 109,
           # 'stool': 110,
           # 'barrel': 111,
           # 'basket': 112,
           # 'waterfall': 113,
           # 'tent': 114,
           # 'bag': 115,
           # 'minibike': 116,
           # 'cradle': 117,
           # 'oven': 118,
           # 'ball': 119,
           # 'food': 120,
           # 'step': 121,
           # 'tank': 122,
           # 'trade name': 123,
           # 'microwave': 124,
           # 'pot': 125,
           # 'animal': 126,
           # 'bicycle': 127,
           # 'lake': 128,
           # 'dishwasher': 129,
           # 'screen': 130,
           # 'blanket': 131,
           # 'sculpture': 132,
           # 'hood': 133,
           # 'sconce': 134,
           # 'vase': 135,
           # 'traffic light': 136,
           # 'tray': 137,
           # 'ashcan': 138,
           # 'fan': 139,
           # 'pier': 140,
           # 'crt screen': 141,
           # 'plate': 142,
           # 'monitor': 143,
           # 'bulletin board': 144,
           # 'shower': 145,
           # 'radiator': 146,
           # 'glass': 147,
           # 'clock': 148,
           # 'flag': 149

