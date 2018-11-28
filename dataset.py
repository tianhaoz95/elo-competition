import os
import pandas as pd

class Dataset():
    def __init__(self):
        self.raw_dataset = {}
    
    def load_raw_dataset(self, dataset_root, dataset_meta):
        abs_train_dataset_filename = os.path.join(dataset_root, dataset_meta['train']['filename'])
        raw_train_dataset_content = pd.read_csv(abs_train_dataset_filename)
        self.raw_dataset['train'] = raw_train_dataset_content