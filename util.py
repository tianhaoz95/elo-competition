import os
import urllib

def create_dir_if_not_exist(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def download_if_not_exist(root_dir, url, filename):
    abs_filename = os.path.join(root_dir, filename)
    create_dir_if_not_exist(root_dir)
    if not os.path.isfile(abs_filename):
        urllib.urlretrieve(url, abs_filename)

def download_dataset():
    print('Downloading dataset ...')

def load_dataset():
    print('Loading dataset ...')