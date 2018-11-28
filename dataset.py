import os
import pandas as pd

class Dataset():
    def __init__(self):
        self.raw_dataset = {}
    
    def load_raw_dataset(self, dataset_root, dataset_meta, dataset_id):
        abs_dataset_filename = os.path.join(dataset_root, dataset_meta[dataset_id]['filename'])
        print('Loading ' + dataset_id + ' from ' + abs_dataset_filename + '...')
        raw_dataset_content = pd.read_csv(abs_dataset_filename)
        self.raw_dataset[dataset_id] = raw_dataset_content