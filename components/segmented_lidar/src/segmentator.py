import numpy as np
from mmseg.apis import inference_model, init_model, show_result_pyplot, MMSegInferencer
import time
import cupy as cp

class Segmentator:
    def __init__(self):

        self.model = MMSegInferencer(model="maskformer_swin-t_upernet_8xb2-160k_ade20k-512x512", device="cuda")
        # Definir la paleta de colores para la segmentación semántica
        self.color_palette = [
            [255, 0, 0],  # wall
            [128, 0, 0],  # building
            [0, 128, 0],  # sky
            [128, 128, 0],  # floor
            [0, 0, 128],  # tree
            [128, 0, 128],  # ceiling
            [0, 128, 128],  # road
            [128, 128, 128],  # bed
            [64, 0, 0],  # windowpane
            [192, 0, 0],  # grass
            [64, 128, 0],  # cabinet
            [192, 128, 0],  # sidewalk
            [64, 0, 128],  # person
            [192, 0, 128],  # earth
            [64, 128, 128],  # door
            [192, 128, 128],  # table
            [0, 64, 0],  # mountain
            [128, 64, 0],  # plant
            [0, 192, 0],  # curtain
            [128, 192, 0],  # chair
            [0, 64, 128],  # car
            [128, 64, 128],  # water
            [0, 192, 128],  # painting
            [128, 192, 128],  # sofa
            [64, 64, 0],  # shelf
            [192, 64, 0],  # house
            [64, 192, 0],  # sea
            [192, 192, 0],  # mirror
            [64, 64, 128],  # rug
            [192, 64, 128],  # field
            [64, 192, 128],  # armchair
            [192, 192, 128],  # seat
            [0, 0, 64],  # fence
            [128, 0, 64],  # desk
            [0, 128, 64],  # rock
            [128, 128, 64],  # wardrobe
            [0, 0, 192],  # lamp
            [128, 0, 192],  # bathtub
            [0, 128, 192],  # railing
            [128, 128, 192],  # cushion
            [64, 0, 64],  # base
            [192, 0, 64],  # box
            [64, 128, 64],  # column
            [192, 128, 64],  # signboard
            [0, 64, 64],  # chest of drawers
            [128, 64, 64],  # counter
            [0, 192, 64],  # sand
            [128, 192, 64],  # sink
            [64, 64, 128],  # skyscraper
            [192, 64, 128],  # fireplace
            [0, 192, 128],  # refrigerator
            [128, 192, 128],  # grandstand
            [64, 0, 192],  # path
            [192, 0, 192],  # stairs
            [0, 128, 192],  # runway
            [128, 128, 192],  # case
            [0, 64, 192],  # pool table
            [128, 64, 192],  # pillow
            [0, 192, 192],  # screen door
            [128, 192, 192],  # stairway
            [64, 64, 0],  # river
            [192, 64, 0],  # bridge
            [0, 192, 0],  # bookcase
            [128, 192, 0],  # blind
            [0, 64, 128],  # coffee table
            [128, 64, 128],  # toilet
            [0, 192, 128],  # flower
            [128, 192, 128],  # book
            [64, 64, 128],  # hill
            [192, 64, 128],  # bench
            [0, 192, 128],  # countertop
            [128, 192, 128],  # stove
            [64, 64, 192],  # palm
            [192, 64, 192],  # kitchen island
            [0, 192, 192],  # computer
            [128, 192, 192],  # swivel chair
            [64, 0, 64],  # boat
            [192, 0, 64],  # bar
            [0, 128, 64],  # arcade machine
            [128, 128, 64],  # hovel
            [0, 64, 192],  # bus
            [128, 64, 192],  # towel
            [0, 192, 192],  # light
            [128, 192, 192],  # truck
            [64, 64, 0],  # tower
            [192, 64, 0],  # chandelier
            [0, 192, 0],  # awning
            [128, 192, 0],  # streetlight
            [0, 64, 128],  # booth
            [128, 64, 128],  # television receiver
            [0, 192, 128],  # airplane
            [128, 192, 128],  # dirt track
            [64, 64, 192],  # apparel
            [192, 64, 192],  # pole
            [0, 192, 192],  # land
            [128, 192, 192],  # bannister
            [64, 64, 0],  # escalator
            [192, 64, 0],  # ottoman
            [0, 192, 0],  # bottle
            [128, 192, 0],  # buffet
            [0, 64, 128],  # poster
            [128, 64, 128],  # stage
            [0, 192, 128],  # van
            [128, 192, 128],  # ship
            [64, 64, 192],  # fountain
            [192, 64, 192],  # conveyer belt
            [0, 192, 192],  # canopy
            [128, 192, 192],  # washer
            [64, 64, 0],  # plaything
            [192, 64, 0],  # swimming pool
            [0, 192, 0],  # stool
            [128, 192, 0],  # barrel
            [0, 64, 128],  # basket
            [128, 64, 128],  # waterfall
            [0, 192, 128],  # tent
            [128, 192, 128],  # bag
            [64, 64, 192],  # minibike
            [192, 64, 192],  # cradle
            [0, 192, 192],  # oven
            [128, 192, 192],  # ball
            [64, 64, 0],  # food
            [192, 64, 0],  # step
            [0, 192, 0],  # tank
            [128, 192, 0],  # trade name
            [0, 64, 128],  # microwave
            [128, 64, 128],  # pot
            [0, 192, 128],  # animal
            [128, 192, 128],  # bicycle
            [64, 64, 192],  # lake
            [192, 64, 192],  # dishwasher
            [0, 192, 192],  # screen
            [128, 192, 192],  # blanket
            [64, 64, 0],  # sculpture
            [192, 64, 0],  # hood
            [0, 192, 0],  # sconce
            [128, 192, 0],  # vase
            [0, 64, 128],  # traffic light
            [128, 64, 128],  # tray
        ]

        # Definir los labels y su inverso
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
        self.inverted_labels = {v: k for k, v in self.labels.items()}

    def extract_points_and_labels_cupy(self, depth_image, segmented_img):
        """Extracts points, categories and labels from depth image and segmentation results using CuPy."""
        # Convert inputs to CuPy arrays if they aren't already
        # depth_image = cp.asarray(depth_image)
        # segmented_img = cp.asarray(segmented_img)

        # Reshape arrays
        points = depth_image.reshape(-1, 3)  # (921600, 3)
        labels = segmented_img.reshape(-1, 1)  # (921600, 1)

        # Create valid mask (non-zero points)
        valid_mask = cp.any(points != 0, axis=1)  # Find non-zero XYZ points

        # Apply mask
        filtered_points = points[valid_mask]  # Shape: (M, 3), M <= 921600
        filtered_labels = labels[valid_mask]  # Shape: (M, 1)

        # Combine points and labels
        result = cp.hstack([filtered_points, filtered_labels])

        # Convert back to numpy if needed (remove if you want to keep results on GPU)
        return result

    def process_frame(self, rgb_frame, depth_frame, img_timestamp):
        result = self.model(rgb_frame, return_datasamples=True)
        segmented_image = cp.asarray(result.pred_sem_seg.data)  # Skip CPU transfer, directly to GPU
        if segmented_image.ndim == 3:
            segmented_image = segmented_image.squeeze(0)  # Now shape [H, W] on GPU
        # segmented_image = result.pred_sem_seg.cpu().data.numpy()
        # if segmented_image.ndim == 3:
        #     segmented_image = segmented_image.squeeze(0)  # Now shape [H, W]
        pointcloud = self.extract_points_and_labels_cupy(depth_frame, segmented_image)
        return pointcloud
