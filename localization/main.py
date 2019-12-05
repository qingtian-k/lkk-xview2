#%%
# Import libraries
# -------------------------------
import os
import sys
from pathlib import Path
sys.path.append(os.path.dirname(__file__))

from model import unet
from data import trainGenerator

import tensorflow as tf
from tensorflow.keras import callbacks, Sequential, Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Input, Reshape, Conv2D, MaxPooling2D, BatchNormalization, Dropout, Activation, Dense, Flatten

print('Tensorflow version:', tf.__version__)

#%% Set variables
# -------------------------------
PATH = Path() / 'data/xBD/spacenet_gt'
DATA_PATH = PATH / 'dataSet'
IMAGES_PATH = PATH / 'images'
LABELS_PATH = PATH / 'labels'
IMAGES_AUG_PATH = PATH / 'aug'

#%%
# Detect hardware
# -------------------------------

default_tpu_name = os.getenv('TPU_NAME')
try:
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver() # TPU detection
except:
    try:
        tpu = tf.distribute.cluster_resolver.TPUClusterResolver(default_tpu_name) # TPU detection
    except ValueError:
        tpu = None
        gpus = tf.config.experimental.list_logical_devices("GPU")

#%%
# Select appropriate distribution strategy
if tpu:
    tf.tpu.experimental.initialize_tpu_system(tpu)
    # Going back and forth between TPU and host is expensive. Better to run 128 batches on the TPU before reporting back.
    strategy = tf.distribute.experimental.TPUStrategy(tpu, steps_per_run=128)
    print('Running on TPU ', tpu.cluster_spec().as_dict()['worker'])
elif len(gpus) > 1:
    strategy = tf.distribute.MirroredStrategy([gpu.name for gpu in gpus])
    print('Running on multiple GPUs ', [gpu.name for gpu in gpus])
elif len(gpus) == 1:
    # default strategy that works on CPU and single GPU
    strategy = tf.distribute.get_strategy()
    print('Running on single GPU ', gpus[0].name)
else:
    # default strategy that works on CPU and single GPU
    strategy = tf.distribute.get_strategy()
    print('Running on CPU')
print("Number of accelerators: ", strategy.num_replicas_in_sync)

#%%
# Create data generator
data_gen_args = dict(rotation_range=45,
                    width_shift_range=0.05,
                    height_shift_range=0.05,
                    shear_range=0.05,
                    zoom_range=0.05,
                    horizontal_flip=True,
                    vertical_flip=True,
                    fill_mode='nearest')
batch_size = 8
print(data_gen_args)

data = trainGenerator(batch_size,PATH,'images','labels',data_gen_args,save_to_dir = None)

#%%
# Make model
# -------------------------------
with strategy.scope():
    model = unet()

# %%    
model.summary()
# %%
type(data)
# %%
steps_per_epoch = 300 // batch_size
epochs = 8
model.fit_generator(data,steps_per_epoch=steps_per_epoch,epochs=epochs)

# %%
