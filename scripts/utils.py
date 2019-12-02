import random
import os
import requests
import gzip
import shutil
import numpy as np
import subprocess

from typing import Any, Union
from pathlib import Path
import tensorflow as tf
# from fastprogress.fastprogress import master_bar, progress_bar

PathOrStr = Union[Path, str]
DEFAULT_DATA_PATH = Path() / 'data'


def ifnone(a: Any, b: Any) -> Any:
    "`a` if `a` is not None, otherwise `b`."
    return b if a is None else a


def download_url(url: str, dest: str, overwrite: bool = False, pbar=None,
                 show_progress=True, chunk_size=1024*1024, timeout=4, retries=5) -> None:
    "Download `url` to `dest` unless it exists and not `overwrite`."
    if os.path.exists(dest) and not overwrite:
        return

    s = requests.Session()
    s.mount('http://', requests.adapters.HTTPAdapter(max_retries=retries))
    u = s.get(url, stream=True, timeout=timeout)
    try:
        file_size = int(u.headers["Content-Length"])
    except:
        show_progress = False

    with open(dest, 'wb') as f:
        nbytes = 0
        # if show_progress: pbar = progress_bar(range(file_size), auto_update=False, leave=False, parent=pbar)
        try:
            for chunk in u.iter_content(chunk_size=chunk_size):
                nbytes += len(chunk)
                # if show_progress: pbar.update(nbytes)
                f.write(chunk)
        except requests.exceptions.ConnectionError as e:
            fname = url.split('/')[-1]
            timeout_txt = (
                '\n Download of {} has failed after {} retries\n'.format(url, retries))
            print(timeout_txt)
            import sys
            sys.exit(1)


def url2name(url): return url.split('/')[-1]


def default_url_fname(url: str):
    return DEFAULT_DATA_PATH / url2name(url)


def download_data(url: str, fname: PathOrStr = None, overwrite=None):
    "Download `url` to destination `fname`."
    fname = Path(ifnone(fname, default_url_fname(url)))
    os.makedirs(fname.parent, exist_ok=True)
    if not fname.exists() or overwrite:
        print('Downloading {} to {}.'.format(url, fname))
        download_url('{}'.format(url), fname, overwrite=overwrite)
    else:
        print('File already exists and was not downloaded. Path: {}'.format(
            fname.absolute()))
    return fname


def untar_file(file, path):
    p1 = 'tar -C {} -zxvf {}'.format(path, file)
    subprocess.run(p1.split(' '))


def unpickle_cifar(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


def adjust_data(data):
    data = data.reshape(-1, 3, 32, 32)
    return np.moveaxis(data, 1, -1)


# ----------------------------------------------------------------------------
# Source: https://www.tensorflow.org/tutorials/load_data/tfrecord
# The following functions can be used to convert a value to a type compatible
# with tf.Example.
# ----------------------------------------------------------------------------
def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        # BytesList won't unpack a string from an EagerTensor.
        value = value.numpy()
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def to_tf_example(img, lab):
    img_raw = img.tostring()
    image_shape = img.shape
    feature = {
        'height': _int64_feature(image_shape[0]),
        'width': _int64_feature(image_shape[1]),
        'depth': _int64_feature(image_shape[2]),
        'label': _int64_feature(lab),
        'image_raw': _bytes_feature(img_raw),
    }
    return tf.train.Example(features=tf.train.Features(feature=feature))


def convert_to_tf_records(input_file, output_file=None, overwrite=False, keep=False):

    if output_file == None:
        output_file = str(input_file) + '.tfrecords'

    if not Path(output_file).exists() or overwrite:
        print('Converting {} to {}.'.format(input_file, output_file))

        data_batch = unpickle_cifar(input_file)
        labels = data_batch[b'labels']
        data = adjust_data(data_batch[b'data'])

        with tf.io.TFRecordWriter(output_file) as writer:
            for i in range(len(labels)):
                img = data[i]
                lab = labels[i]
                tf_example = to_tf_example(img, lab)
                writer.write(tf_example.SerializeToString())
            if keep == False:
                os.remove(input_file)

    else:
        print('File already exists and was not overwriten. Path: {}'.format(output_file))
    return output_file


def sync_data_to_bucket(bucket):
    p1 = 'mkdir -p data'
    p2 = 'gsutil -m rsync -r data {}/data'.format(bucket)
    subprocess.run(p1.split(' '))
    subprocess.run(p2.split(' '))
