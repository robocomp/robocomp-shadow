''' Installation instructions


'''

## Runs with pytorch 1.10.1
import torch
import sys
from PIL import Image
from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation
import time
#from pynvml import *
import cv2
import numpy as np

class Mask2Former():
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

    def draw_semantic_segmentation(self, seg, color_image):
        color_seg = np.zeros((seg.shape[0], seg.shape[1], 3), dtype=np.uint8)  # height, width, 3
        palette = np.array(self.color_palette)
        for label, color in enumerate(palette):
            color_seg[seg == label, :] = color
        # Convert to BGR
        color_seg = color_seg[..., ::-1]

        # Show image + mask
        img = np.array(color_image) * 0.5 + color_seg * 0.5
        img = img.astype(np.uint8)
        cv2.imshow("", np.asarray(img))
        cv2.waitKey(2)

    def create_mask(self, seg):
        mask = np.zeros((seg.shape[0], seg.shape[1], 1), dtype=np.uint8)  # height, width, 3
        mask[(seg == 3) | (seg == 91) | (seg == 52), :] = 255
        # labels_ids = torch.unique(seg).tolist()
        # for label_id in labels_ids:
        #     label = self.model.config.id2label[label_id]
        #     print(label, label_id)
        return mask

    def process(self, img):
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
        #segmented_img = predicted_semantic_map.cpu()
        instance_img = self.processor.post_process_instance_segmentation(outputs)[0]
        #print(instance_segmentation['segments_info'])
        #self.draw_semantic_segmentation(predicted_semantic_map.cpu(), img)
        mask = self.create_mask(segmented_img)
        return mask, segmented_img, instance_img



 # self.labels = {"building": 31,
 #                      "sky": 2,
 #                      "floor": 3,
 #                      "tree": 4,
 #                      "earth": 13,
 #                      "mountain": 16,
 #                      "car": 20,
 #                      "field": 29,
 #                      "fence": 32,
 #                      "path": 52,
 #                      "truck": 83,
 #                      "dirt track": 91,
 #                      "wall": 0,
 #                      "ceiling": 5,
 #                      "windowpane": 8,
 #                      "person": 12,
 #                      "door": 14,
 #                      "railing": 38,
 #                      "signboard": 43,
 #                      "light": 82,
 #                      "plaything": 108,
 #                      "sconce": 134,
 #                      "ashcan": 138
 #                    }