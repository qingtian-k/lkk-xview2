#%% Import libraries
# -------------------------------
from pathlib import Path
import numpy as np
import imageio

#%% Set variables
# -------------------------------
PATH = Path('/home/lucaskawazoi/lkk-xview2/data/xBD/spacenet_gt')
LABELS_PATH = PATH / 'labels'
GRAYSCALE_LABELS_PATH = PATH / 'grayscale_labels'
GRAYSCALE_LABELS_PATH.mkdir(exist_ok=True)

#%%
labels_list = sorted(LABELS_PATH.iterdir())

for label in labels_list:
    new_label = GRAYSCALE_LABELS_PATH / label.name

    # Check : All channels are the same
    # np.all(imageio.imread(label)[:,:,0] == imageio.imread(label)[:,:,2])
    gray_scale_img = imageio.imread(label)[:,:,0]
    imageio.imsave(new_label, gray_scale_img)