from fastai.vision.all import *
from fastai.text.all import *
from fastai.collab import *
from fastai.tabular.all import *
import numpy as np
from pathlib import Path
#torch.backends.cudnn.benchmark=True

path = Path('/home/pbustos/Software/python/Road-surface-detection-and-differentiation-considering-surface-damages')

codes = np.loadtxt(path/'codes.txt', dtype=str)
print("Codes", codes)

path_img = path/'images'
fnames = get_image_files(path_img)
print("Images", len(fnames))

path_lbl = path/'labels'
lbl_names = get_image_files(path_lbl)
print("Labels", len(lbl_names))

def label_func(fn): return path/"labels"/f"{fn.stem}GT{fn.suffix}"

dls = SegmentationDataLoaders.from_label_func(path=path, bs=8, fnames = fnames, codes = codes, label_func=label_func)

dls.show_batch(max_n=6)

learn = unet_learner(dls, resnet34)
learn.fine_tune(6)