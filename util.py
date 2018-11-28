import os
import urllib
import config
import dataset

def create_dir_if_not_exist(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def download_if_not_exist(root_dir, url, filename):
    abs_filename = os.path.join(root_dir, filename)
    create_dir_if_not_exist(root_dir)
    if not os.path.isfile(abs_filename):
        urllib.request.urlretrieve(url, abs_filename)

def download_dataset():
    print('Downloading dataset ...')
    dataset_file_cnt = len(config.dataset_filenames)
    print('Found ' + str(dataset_file_cnt) + ' files in dataset')
    for meta in config.dataset_meta:
        print('Fetching ' + meta['filename'] + ' from ' + meta['download_url'])
        download_if_not_exist(config.dataset_root_dir, meta['download_url'], meta['filename'])

def load_dataset():
    print('Loading dataset ...')
    res = dataset.Dataset()
    res.load_raw_dataset(config.dataset_root_dir, config.dataset_filenames)
    return res