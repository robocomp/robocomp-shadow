## Runs with pytorch 1.10.1
import torch
from PIL import Image
from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation
import time
from pynvml import *
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import cm

fig, ax = plt.subplots()

def print_gpu_utilization():
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(0)
    info = nvmlDeviceGetMemoryInfo(handle)
    proc = nvmlDeviceGetPowerUsage(handle)
    print(f"GPU memory occupied: {info.used//1024**2} MB.", " Power ", proc)

def draw_semantic_segmentation(segmentation):
    global fig, ax
    # get the used color map
    viridis = cm.get_cmap('viridis', torch.max(segmentation))
    # get all the unique numbers
    labels_ids = torch.unique(segmentation).tolist()
    ax.imshow(segmentation)
    handles = []
    for label_id in labels_ids:
        label = model.config.id2label[label_id]
        color = viridis(label_id)
        handles.append(mpatches.Patch(color=color, label=label))
    ax.legend(handles=handles)
    plt.draw()
    plt.pause(0.001)
    return fig

def draw_semantic_segmentation2(seg, color_image):
    color_seg = np.zeros((seg.shape[0], seg.shape[1], 3), dtype=np.uint8)  # height, width, 3
    palette = np.array(color_palette)
    for label, color in enumerate(palette):
        color_seg[seg == label, :] = color
    # Convert to BGR
    color_seg = color_seg[..., ::-1]

    # Show image + mask
    img = np.array(color_image) * 0.5 + color_seg * 0.5
    img = img.astype(np.uint8)
    cv2.imshow("", np.asarray(img))
    cv2.waitKey(2)

# load Mask2Former fine-tuned on ADE20k semantic segmentation
processor = AutoImageProcessor.from_pretrained("facebook/mask2former-swin-tiny-ade-semantic")
model = Mask2FormerForUniversalSegmentation.from_pretrained("facebook/mask2former-swin-tiny-ade-semantic")
color_palette = [list(np.random.choice(range(256), size=3)) for _ in range(len(model.config.id2label))]
model = model.to('cuda:0')

# Read until video is completed
cap = cv2.VideoCapture('jardin_3.mp4')
while (cap.isOpened()):
    ret, img = cap.read()
    if ret:
        start = time.time()
        img = cv2.resize(img, (500, 500))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        #cv2.imshow("Trans", img)
        #cv2.waitKey(2)
        im_pil = Image.fromarray(img)
        inputs = processor(images=im_pil, return_tensors="pt").to('cuda:0')


        with torch.no_grad():
            outputs = model(**inputs)
        #print_gpu_utilization()

        # model predicts class_queries_logits of shape `(batch_size, num_queries)`
        # and masks_queries_logits of shape `(batch_size, num_queries, height, width)`
        class_queries_logits = outputs.class_queries_logits
        masks_queries_logits = outputs.masks_queries_logits

        # you can pass them to processor for postprocessing
        predicted_semantic_map = processor.post_process_semantic_segmentation(outputs, target_sizes=[im_pil.size[::-1]])[0]
        # we refer to the demo notebooks for visualization (see "Resources" section in the Mask2Former docs)
        draw_semantic_segmentation2(predicted_semantic_map.cpu(), img)
        print(time.time()-start)




