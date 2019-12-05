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
folder = '/home/lucaskawazoi/lkk-xview2/data/xBD/spacenet_gt/images/*.png'
print(len(glob.glob(folder)[:5]))

#%%
for image_path in glob.glob(folder)[:5]:
    image = imageio.imread(image_path)
    print(image.shape)
    print(image.dtype)

# %%
list_folder = '/home/lucaskawazoi/lkk-xview2/data/xBD/spacenet_gt/dataSet'
dataset_splits = tf.gfile.Glob(os.path.join(list_folder, '*.txt'))

# %%
dataset_split = dataset_splits[0]
dataset_split

# %%
dataset = os.path.basename(dataset_split)[:-4]
dataset

# %%
sys.stdout.write('Processing ' + dataset)

# %%
filenames = [x.strip('\n') for x in open(dataset_split, 'r')]
filenames
# %%
num_images = len(filenames)
num_images
# %%
num_per_shard = int(math.ceil(num_images / _NUM_SHARDS))
num_per_shard

# %%
image_reader = build_data.ImageReader('png', channels=3)
image_reader

# %%
label_reader = build_data.ImageReader('png', channels=3)
label_reader