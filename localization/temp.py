#%% Import libraries
# -------------------------------
import os
import sys
from pathlib import Path
sys.path.append(os.path.dirname(__file__))

import numpy as np
import imageio
import glob
import tensorflow as tf
import math
import build_data

_NUM_SHARDS = 4

#%% Set variables
# -------------------------------
folder = '/home/lucaskawazoi/lkk-xview2/data/xBD/spacenet_gt/labels/*.png'
f = glob.glob(folder)[0]
f

#%%
np.unique(imageio.imread(f)[:,:,2])


#%%
folder = '/home/lucaskawazoi/models/research/deeplab/datasets/pascal_voc_seg/VOCdevkit/VOC2012/SegmentationClassRaw/*.png'
f = glob.glob(folder)[0]
f

# %%
imageio.imread(f).shape