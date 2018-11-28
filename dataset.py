import os
import pandas as pd

class Dataset():
    def __init__(self):
        self.raw_dataset = {}
    
    def load_raw_dataset(self, dataset_root, dataset_meta):
        for meta in dataset_meta:
            abs_dataset_filename = os.path.join(dataset_root, meta['filename'])
            print('Loading dataset from ' + meta['filename'] + '...')
            raw_dataset_content = pd.read_csv(abs_dataset_filename)
            self.raw_dataset[meta['filename']] = raw_dataset_content