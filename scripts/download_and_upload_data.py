# Check [notebook download_and_upload_data.ipynb](/nbs/download_and_upload_data.ipynb) 
# to see details on how I got these functions, specially the convert_to_tf_records

# --------------------------------------------------------------
# IMPORT LIBRARIES
# --------------------------------------------------------------
import subprocess
from absl import flags, app
from pathlib import Path
from utils import url2name, download_data, untar_file, unpickle_cifar, convert_to_tf_records, sync_data_to_bucket
import matplotlib.pyplot as plt

# --------------------------------------------------------------
# SET VARIABLES
# --------------------------------------------------------------
flags.DEFINE_string('dest_dir', None, 'Destination for uploading the dataset')
flags.DEFINE_string('DATA_PATH', './data', 'local path where dataset will be stored') # for large datasets, should mark it as optional with default False
FLAGS = flags.FLAGS

URLS = ['https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz']

def main(argv):
    DATA_PATH = Path(FLAGS.DATA_PATH)

    # --------------------------------------------------------------
    # DOWNLOAD DATA
    # --------------------------------------------------------------
    for url in URLS:
        download_data(url=url, fname=DATA_PATH/url2name(url))

    # --------------------------------------------------------------
    # UNTAR GZ FILES
    # --------------------------------------------------------------
    files = list(sorted(DATA_PATH.rglob('*.gz')))
    for f in files:
        untar_file(f, DATA_PATH)

    # --------------------------------------------------------------
    # CONVERT THE RAW DATA INTO TF-RECORDS
    # --------------------------------------------------------------
    files = list(sorted(DATA_PATH.rglob('*batch_[0-9]'))) + \
        list(sorted(DATA_PATH.rglob('*test_batch')))
    for f in files:
        convert_to_tf_records(f, overwrite=True)

    # --------------------------------------------------------------
    # SYNC DATA TO BUCKET
    # --------------------------------------------------------------
    sync_data_to_bucket(FLAGS.dest_dir)


if __name__ == '__main__':
    app.run(main)