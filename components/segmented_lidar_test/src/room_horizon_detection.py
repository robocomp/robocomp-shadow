import os
import sys
import glob
import json
import argparse
import numpy as np
from PIL import Image
from tensorboard.compat.tensorflow_stub.dtypes import uint64
from tqdm import tqdm
from scipy.ndimage.filters import maximum_filter
from shapely.geometry import Polygon

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.append("../HorizonNet")

from HorizonNet.model import HorizonNet
from HorizonNet.dataset import visualize_a_data
from HorizonNet.misc import post_proc, panostretch, utils
from HorizonNet.eval_general import layout_2_depth

class RoomHorizonDetection:
    def __init__(self):

        self.device = torch.device('cuda')

        # Loaded trained model
        self.net = utils.load_trained_model(HorizonNet, "HorizonNet/checkpoints/resnet50_rnn__mp3d.pth").to(self.device)
        self.net.eval()

    def infer_possible_corners(self, image):
        # Convert from np.array to PIL Image
        if isinstance(image, np.ndarray):
            img_pil = Image.fromarray(image.astype(np.uint8))
        else:
            img_pil = image
        # Store original image size
        original_size = img_pil.size
        if img_pil.size != (1024, 512):
            img_pil = img_pil.resize((1024, 512), Image.BICUBIC)
        img_ori = np.array(img_pil)[..., :3].transpose([2, 0, 1]).copy()
        x = torch.FloatTensor([img_ori / 255])

        # Get W and H
        # print("Image shape:", image.shape)
        H, W, _ = image.shape

        # Inferenceing corners
        return self.inference(net=self.net, x=x, device=self.device, visualize=True, r=0.05)

    # def infer_corners(self, image):
    #     # Convert from np.array to PIL Image
    #     if isinstance(image, np.ndarray):
    #         img_pil = Image.fromarray(image.astype(np.uint8))
    #     else:
    #         img_pil = image
    #     # Store original image size
    #     original_size = img_pil.size
    #     if img_pil.size != (1024, 512):
    #         img_pil = img_pil.resize((1024, 512), Image.BICUBIC)
    #     img_ori = np.array(img_pil)[..., :3].transpose([2, 0, 1]).copy()
    #     x = torch.FloatTensor([img_ori / 255])
    #
    #     # Get W and H
    #     # print("Image shape:", image.shape)
    #     H, W, _ = image.shape
    #
    #     # Inferenceing corners
    #     cor_id, vis_out = self.inference(net=self.net, x=x, device=self.device, visualize=True, r=0.05)
    #     # Transform points to original image size
    #     cor_id = np.array(cor_id) * (W / 1024, H / 512)
    #     cor_id = cor_id.astype(np.float32)
    #     return cor_id, vis_out

    def find_N_peaks(self, signal, r=29, min_v=0.05, N=None):
        max_v = maximum_filter(signal, size=r, mode='wrap')
        pk_loc = np.where(max_v == signal)[0]
        pk_loc = pk_loc[signal[pk_loc] > min_v]
        if N is not None:
            order = np.argsort(-signal[pk_loc])
            pk_loc = pk_loc[order[:N]]
            pk_loc = pk_loc[np.argsort(pk_loc)]

        return pk_loc, signal[pk_loc]


    def augment(self, x_img, flip, rotate):
        x_img = x_img.numpy()
        aug_type = ['']
        x_imgs_augmented = [x_img]
        if flip:
            aug_type.append('flip')
            x_imgs_augmented.append(np.flip(x_img, axis=-1))
        for shift_p in rotate:
            shift = int(round(shift_p * x_img.shape[-1]))
            aug_type.append('rotate %d' % shift)
            x_imgs_augmented.append(np.roll(x_img, shift, axis=-1))
        return torch.FloatTensor(np.concatenate(x_imgs_augmented, 0)), aug_type


    def augment_undo(self, x_imgs_augmented, aug_type):
        x_imgs_augmented = x_imgs_augmented.detach().cpu().numpy()
        sz = x_imgs_augmented.shape[0] // len(aug_type)
        x_imgs = []
        for i, aug in enumerate(aug_type):
            x_img = x_imgs_augmented[i*sz : (i+1)*sz]
            if aug == 'flip':
                x_imgs.append(np.flip(x_img, axis=-1))
            elif aug.startswith('rotate'):
                shift = int(aug.split()[-1])
                x_imgs.append(np.roll(x_img, -shift, axis=-1))
            elif aug == '':
                x_imgs.append(x_img)
            else:
                raise NotImplementedError()

        return np.array(x_imgs)

    def inference(self, net, x, device, flip=False, rotate=[], visualize=False,
                  force_cuboid=False, force_raw=False, min_v=None, r=0.05):
        '''
        net   : the trained HorizonNet
        x     : tensor in shape [1, 3, 512, 1024]
        flip  : fliping testing augmentation
        rotate: horizontal rotation testing augmentation
        '''

        H, W = tuple(x.shape[2:])

        # Network feedforward (with testing augmentation)
        x, aug_type = self.augment(x, flip, rotate)
        y_bon_, y_cor_ = net(x.to(device))
        y_bon_ = self.augment_undo(y_bon_.cpu(), aug_type).mean(0)
        y_cor_ = self.augment_undo(torch.sigmoid(y_cor_).cpu(), aug_type).mean(0)

        # Visualize raw model output
        if visualize:
            vis_out = visualize_a_data(x[0],
                                       torch.FloatTensor(y_bon_[0]),
                                       torch.FloatTensor(y_cor_[0]))
        else:
            vis_out = None

        y_bon_ = (y_bon_[0] / np.pi + 0.5) * H - 0.5
        y_bon_[0] = np.clip(y_bon_[0], 1, H / 2 - 1)
        y_bon_[1] = np.clip(y_bon_[1], H / 2 + 1, H - 2)
        y_cor_ = y_cor_[0, 0]


        # Detech wall-wall peaks
        if min_v is None:
            min_v = 0 if force_cuboid else 0.05

        r = int(round(W * r / 2))
        N = 4 if force_cuboid else None
        xs_ = self.find_N_peaks(y_cor_, r=r, min_v=min_v, N=N)[0]

        # Get peak angle considering image horizontal values (W/2 values is 0)
        corners_angles = xs_ / W * np.pi - np.pi / 2

        return -corners_angles, vis_out

    # def inference(self, net, x, device, flip=False, rotate=[], visualize=False,
    #               force_cuboid=False, force_raw=False, min_v=None, r=0.05):
    #     '''
    #     net   : the trained HorizonNet
    #     x     : tensor in shape [1, 3, 512, 1024]
    #     flip  : fliping testing augmentation
    #     rotate: horizontal rotation testing augmentation
    #     '''
    #
    #     H, W = tuple(x.shape[2:])
    #
    #     # Network feedforward (with testing augmentation)
    #     x, aug_type = self.augment(x, flip, rotate)
    #     y_bon_, y_cor_ = net(x.to(device))
    #     y_bon_ = self.augment_undo(y_bon_.cpu(), aug_type).mean(0)
    #     y_cor_ = self.augment_undo(torch.sigmoid(y_cor_).cpu(), aug_type).mean(0)
    #
    #     # Visualize raw model output
    #     if visualize:
    #         vis_out = visualize_a_data(x[0],
    #                                    torch.FloatTensor(y_bon_[0]),
    #                                    torch.FloatTensor(y_cor_[0]))
    #     else:
    #         vis_out = None
    #
    #     y_bon_ = (y_bon_[0] / np.pi + 0.5) * H - 0.5
    #     y_bon_[0] = np.clip(y_bon_[0], 1, H / 2 - 1)
    #     y_bon_[1] = np.clip(y_bon_[1], H / 2 + 1, H - 2)
    #     y_cor_ = y_cor_[0, 0]
    #
    #
    #     # Detech wall-wall peaks
    #     if min_v is None:
    #         min_v = 0 if force_cuboid else 0.05
    #
    #     r = int(round(W * r / 2))
    #     N = 4 if force_cuboid else None
    #     xs_ = self.find_N_peaks(y_cor_, r=r, min_v=min_v, N=N)[0]
    #
    #     # Print detected peaks
    #     print('Detected peaks:', xs_)
    #
    #     # Considering xs_ stores the x coordinates of the detected peaks,
    #     # I want to store the (x, y) coordinates that delimits the horizon
    #     # Between each pair of peaks.
    #     # y_bon_ is a 2D array with shape (2, W) where the first row is the upper bound
    #     # and the second row is the lower bound of the horizon in height coordinates.
    #
    #     y_bon__ = y_bon_.T
    #
    #     puntos_esquinas = []
    #     for i, x in enumerate(xs_):
    #         y_techo, y_suelo = y_bon__[x]
    #         puntos_esquinas.append(np.array([x, y_suelo]))
    #         puntos_esquinas.append(np.array([x, y_techo]))
    #
    #     # Como resultado tienes una lista con puntos (x,y) que delimitan cada esquina
    #     puntos_esquinas = np.array(puntos_esquinas)  # shape (2*N, 2)
    #
    #     return puntos_esquinas, vis_out

