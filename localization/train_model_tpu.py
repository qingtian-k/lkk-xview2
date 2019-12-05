#%%
# -------------------------------
# Import libraries
# -------------------------------
import os
import sys
sys.path.append(os.path.dirname(__file__))

from model import *
# from utils import load_data

import tensorflow as tf
from tensorflow.keras import callbacks, Sequential
from tensorflow.keras.layers import Reshape, Conv2D, MaxPooling2D, BatchNormalization, Dropout, Activation, Dense, Flatten
print('Tensorflow version:', tf.__version__)

#%%
# -------------------------------
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
# -------------------------------
# Set variables
# -------------------------------
batchsize = 16 * strategy.num_replicas_in_sync
test_batchsize = 4 * strategy.num_replicas_in_sync
epoch = 50
frequency = 1
out = 'logs'
resume = ''
tcrop = 400
vcrop = 480

print('# Minibatch-size: {}'.format(batchsize))
print('# Crop-size: {}'.format(tcrop))
print('# epoch: {}'.format(epoch))
print('')

LEARNING_RATE = 0.01
LEARNING_RATE_EXP_DECAY = 0.6 if strategy.num_replicas_in_sync == 1 else 0.7

#%%
# # -------------------------------
# # Make model
# # -------------------------------
model = unet()


#%%
# # -------------------------------
# # Make dataset
# # -------------------------------

# # load your data
# x_train, y_train, x_val, y_val = load_data(...)

# # preprocess input
# x_train = preprocess_input(x_train)
# x_val = preprocess_input(x_val)

#%%
# # -------------------------------
# # Make model
# # -------------------------------
# def make_model():
#     model = Sequential([
#         Reshape(input_shape=(28*28,), target_shape=(28, 28, 1), name='image'),

#         Conv2D(filters=12, kernel_size=3, padding='same', use_bias=False),
#         BatchNormalization(scale=False, center=True),
#         Activation('relu'),

#         Conv2D(filters=24, kernel_size=6, padding='same',
#                use_bias=False, strides=2),
#         BatchNormalization(scale=False, center=True),
#         Activation('relu'),

#         Conv2D(filters=36, kernel_size=6, padding='same',
#                use_bias=False, strides=2),
#         BatchNormalization(scale=False, center=True),
#         Activation('relu'),

#         Flatten(),
#         Dense(2000, use_bias=False),
#         BatchNormalization(scale=False, center=True),
#         Activation('relu'),
#         Dropout(0.4),
#         Dense(10, activation='softmax')


#     ])

#     model.compile(optimizer='adam',
#                   loss='categorical_crossentropy',
#                   metrics=['accuracy'])

#     return model


# with strategy.scope():
#     model = make_model()

# model.summary()

# # -------------------------------
# # Learning rate decay
# # -------------------------------
# lr_decay = callbacks.LearningRateScheduler(
#     lambda epoch: LEARNING_RATE * LEARNING_RATE_EXP_DECAY**epoch,
#     verbose=True)


# # -------------------------------
# # MODEL.FIT
# # -------------------------------
# EPOCHS = 10
# steps_per_epoch = 60000 // BATCH_SIZE
# history = model.fit(training_dataset,
#                     steps_per_epoch=steps_per_epoch,
#                     epochs=EPOCHS,
#                     callbacks=[lr_decay]
#                     )

# final_stats = model.evaluate(validation_dataset, steps=1)
# print('Validation accuracy: ',final_stats[1])