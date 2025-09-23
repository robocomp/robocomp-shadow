import numpy as np
from mmseg.apis.inference import inference_model, init_model, show_result_pyplot
from mmseg.apis.mmseg_inferencer import MMSegInferencer

import time
import cupy as cp
import cv2
from cupyx.scipy.ndimage import binary_dilation


class Segmentator:
    def __init__(self):

        # self.model = MMSegInferencer(model="deeplabv3plus_r101-d8_4xb4-160k_ade20k-512x512", device="cuda")
        self.model = MMSegInferencer(model="resnest_s101-d8_deeplabv3plus_4xb4-160k_ade20k-512x512", device="cuda")
        # self.depth_model = MMSegInferencer(model="vpd_sd_4xb8-25k_nyu-512x512", device="cuda")
        # Definir la paleta de colores para la segmentación semántica
        self.color_palette = np.array(
            [[120, 120, 120], [180, 120, 120], [6, 230, 230], [80, 50, 50],
             [4, 200, 3], [120, 120, 80], [140, 140, 140], [204, 5, 255],
             [230, 230, 230], [4, 250, 7], [224, 5, 255], [235, 255, 7],
             [150, 5, 61], [120, 120, 70], [8, 255, 51], [255, 6, 82],
             [143, 255, 140], [204, 255, 4], [255, 51, 7], [204, 70, 3],
             [0, 102, 200], [61, 230, 250], [255, 6, 51], [11, 102, 255],
             [255, 7, 71], [255, 9, 224], [9, 7, 230], [220, 220, 220],
             [255, 9, 92], [112, 9, 255], [8, 255, 214], [7, 255, 224],
             [255, 184, 6], [10, 255, 71], [255, 41, 10], [7, 255, 255],
             [224, 255, 8], [102, 8, 255], [255, 61, 6], [255, 194, 7],
             [255, 122, 8], [0, 255, 20], [255, 8, 41], [255, 5, 153],
             [6, 51, 255], [235, 12, 255], [160, 150, 20], [0, 163, 255],
             [140, 140, 140], [250, 10, 15], [20, 255, 0], [31, 255, 0],
             [255, 31, 0], [255, 224, 0], [153, 255, 0], [0, 0, 255],
             [255, 71, 0], [0, 235, 255], [0, 173, 255], [31, 0, 255],
             [11, 200, 200], [255, 82, 0], [0, 255, 245], [0, 61, 255],
             [0, 255, 112], [0, 255, 133], [255, 0, 0], [255, 163, 0],
             [255, 102, 0], [194, 255, 0], [0, 143, 255], [51, 255, 0],
             [0, 82, 255], [0, 255, 41], [0, 255, 173], [10, 0, 255],
             [173, 255, 0], [0, 255, 153], [255, 92, 0], [255, 0, 255],
             [255, 0, 245], [255, 0, 102], [255, 173, 0], [255, 0, 20],
             [255, 184, 184], [0, 31, 255], [0, 255, 61], [0, 71, 255],
             [255, 0, 204], [0, 255, 194], [0, 255, 82], [0, 10, 255],
             [0, 112, 255], [51, 0, 255], [0, 194, 255], [0, 122, 255],
             [0, 255, 163], [255, 153, 0], [0, 255, 10], [255, 112, 0],
             [143, 255, 0], [82, 0, 255], [163, 255, 0], [255, 235, 0],
             [8, 184, 170], [133, 0, 255], [0, 255, 92], [184, 0, 255],
             [255, 0, 31], [0, 184, 255], [0, 214, 255], [255, 0, 112],
             [92, 255, 0], [0, 224, 255], [112, 224, 255], [70, 184, 160],
             [163, 0, 255], [153, 0, 255], [71, 255, 0], [255, 0, 163],
             [255, 204, 0], [255, 0, 143], [0, 255, 235], [133, 255, 0],
             [255, 0, 235], [245, 0, 255], [255, 0, 122], [255, 245, 0],
             [10, 190, 212], [214, 255, 0], [0, 204, 255], [20, 0, 255],
             [255, 255, 0], [0, 153, 255], [0, 41, 255], [0, 255, 204],
             [41, 0, 255], [41, 255, 0], [173, 0, 255], [0, 245, 255],
             [71, 0, 255], [122, 0, 255], [0, 255, 184], [0, 92, 255],
             [184, 255, 0], [0, 133, 255], [255, 214, 0], [25, 194, 194],
             [102, 255, 0], [92, 0, 255]])

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


    def extract_points_and_labels_cupy(self, depth_image, segmented_img):
        """Extracts points, categories and labels from depth image and segmentation results using CuPy."""
        # Convert inputs to CuPy arrays if they aren't already
        # depth_image = cp.asarray(depth_image)
        # segmented_img = cp.asarray(segmented_img)

        # Reshape arrays
        points = depth_image.reshape(-1, 3)  # (921600, 3)
        labels = segmented_img.reshape(-1, 1)  # (921600, 1)

        # Create valid mask (non-zero points)
        # valid_mask = cp.any(points != 0, axis=1)  # Find non-zero XYZ points
        valid_mask = np.any(points != 0, axis=1) & (labels.squeeze() != 255)

        # Apply mask
        filtered_points = points[valid_mask]  # Shape: (M, 3), M <= 921600
        filtered_labels = labels[valid_mask]  # Shape: (M, 1)

        # Combine points and labels
        result = np.hstack([filtered_points, filtered_labels])

        # Convert back to numpy if needed (remove if you want to keep results on GPU)
        return result

    def process_frame(self, rgb_frame, depth_frame, img_timestamp):
        result = self.model(rgb_frame, return_datasamples=True)
        segmented_image = result.pred_sem_seg.data.cpu().numpy()
        if segmented_image.ndim == 3:
            segmented_image = segmented_image.squeeze(0)  # Now shape [H, W] on GPU
        # segmented_image = self.add_zero_borders_to_segmentation_gpu(segmented_image, 15)
        pointcloud = self.extract_points_and_labels_cupy(depth_frame, segmented_image)
        return pointcloud, segmented_image



    def add_zero_borders_to_segmentation_gpu(self, mask_gpu, kernel_size=3):
        """
        Args:
            mask: Máscara de segmentación (2D) en CPU (numpy.ndarray) o GPU (cupy.ndarray).
            kernel_size: Tamaño del kernel para la operación morfológica (default=3).
        Returns:
            Máscara modificada con ceros en los bordes entre clases (en GPU como cupy.ndarray).
        """
        # Inicializar máscara de bordes en GPU
        border_mask_gpu = cp.zeros_like(mask_gpu, dtype=bool)

        # Estructura de conectividad (kernel de cruz para 4-vecinos)
        kernel = cp.ones((kernel_size, kernel_size), dtype=bool)
        if kernel_size == 3:
            kernel[1, 1] = False  # Para conectividad 4-vecinos (cruz)

        # Procesar cada clase única
        unique_classes = cp.unique(mask_gpu)

        for cls in unique_classes:
            # Máscara binaria para la clase actual
            class_mask = (mask_gpu == cls)

            # Dilatación binaria en GPU
            dilated = binary_dilation(class_mask, structure=kernel)

            # Bordes externos de la clase
            border_pixels = dilated & ~class_mask

            # Acumular bordes
            border_mask_gpu |= border_pixels

        # Aplicar bordes a la máscara original
        result_gpu = mask_gpu.copy()
        result_gpu[border_mask_gpu] = 255

        return result_gpu

    # def mask_to_color(self, mask):
    #
    #     # Inicializar la imagen en color
    #     h, w = mask.shape
    #     color_image = np.zeros((h, w, 3), dtype=np.uint8)
    #
    #     # Para cada categoría en la máscara, aplicar su color
    #     for category in np.unique(mask):
    #
    #         if category >= len(self.color_palette):
    #             continue
    #         color_image[mask == category] = self.color_palette[category]
    #     return color_image

    def mask_to_color(self, mask):
        # Inicializar la imagen en color
        h, w = mask.shape
        color_image = np.zeros((h, w, 3), dtype=np.uint8)

        # Para cada categoría en la máscara, aplicar su color
        unique_categories = np.unique(mask)
        for category in unique_categories:
            if category >= len(self.color_palette):
                continue
            color_image[mask == category] = self.color_palette[category]

        # --- Dibujar leyenda ---
        legend_height = 25
        legend_width = 180
        margin = 10
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1

        # Solo mostrar hasta 15 categorías para no saturar la imagen
        max_legend = 20
        for idx, category in enumerate(unique_categories[:max_legend]):
            if category >= len(self.color_palette):
                continue
            color = [int(c) for c in self.color_palette[category]]
            label = list(self.labels.keys())[list(self.labels.values()).index(category)] \
                if category in self.labels.values() else f"Clase {category}"

            legend_text = f"{label} ({category})"

            y0 = h - margin - (idx + 1) * legend_height
            x0 = w - legend_width - margin
            y1 = y0 + legend_height - 5
            x1 = x0 + 20

            # Rectángulo de color
            cv2.rectangle(color_image, (x0, y0), (x1, y1), color, -1)
            # Texto de la categoría
            cv2.putText(color_image, legend_text, (x1 + 5, y1 - 5), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

        return color_image

    def estimate_depth(self, image):
        """
        Estima un mapa de profundidad monocular a partir de una imagen RGB,
        utilizando MMSegInferencer con un modelo como AdaBins o VPD.

        Args:
            image (np.ndarray): Imagen RGB en formato (H, W, 3), valores uint8.

        Returns:
            depth_map (np.ndarray): Mapa de profundidad normalizado a float32.
        """
        result = self.depth_model(image, return_datasamples=True)
        depth_tensor = result.pred_depth_map.data  # Tensor (1, H, W)
        depth = depth_tensor.squeeze().cpu().numpy()  # Convertir a (H, W)

        # Normalizar profundidad entre 0 y 1 para visualización o procesamiento
        depth_normalized = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
        return depth_normalized

    def visualize_depth(self, depth_map, colormap=cv2.COLORMAP_MAGMA):
        """Convierte un mapa de profundidad a una imagen visualizable (color)."""
        depth_8bit = (depth_map * 255).astype(np.uint8)
        depth_colored = cv2.applyColorMap(depth_8bit, colormap)
        return depth_colored