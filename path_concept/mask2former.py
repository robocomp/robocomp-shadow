''' Installation instructions


'''

## Runs with pytorch 1.10.1
import torch
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
        self.labels = {"building": 31,
                      "sky": 2,
                      "floor": 3,
                      "tree": 4,
                      "earth": 13,
                      "mountain": 16,
                      "car": 20,
                      "field": 29,
                      "fence": 32,
                      "path": 52,
                      "truck": 83,
                      "dirt track": 91}

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

    def create_mask(self, seg, color_image):
        mask = np.zeros((seg.shape[0], seg.shape[1], 1), dtype=np.uint8)  # height, width, 3
        mask[seg == 3, :] = 255
        mask[seg == 91, :] = 255
        mask[seg == 52, :] = 255
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
        predicted_semantic_map = self.processor.post_process_semantic_segmentation(outputs, target_sizes=[im_pil.size[::-1]])[0]
        #self.draw_semantic_segmentation(predicted_semantic_map.cpu(), img)
        mask = self.create_mask(predicted_semantic_map.cpu(), img)
        return mask

