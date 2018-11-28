import os
import urllib
import config

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
    for i in range(dataset_file_cnt):
        print('Fetching ' + config.dataset_filenames[i] + ' from ' + config.dataset_download_urls[i])
        download_if_not_exist(config.dataset_root_dir, config.dataset_download_urls[i], config.dataset_filenames[i])

def load_dataset():
    print('Loading dataset ...')