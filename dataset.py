import os
import pandas as pd

class Dataset():
    def __init__(self):
        self.raw_dataset = {}
    
    def load_raw_dataset(self, dataset_root, dataset_filenames):
        for dataset_filename in dataset_filenames:
            abs_dataset_filename = os.path.join(dataset_root, dataset_filename)
            raw_dataset_content = pd.read_csv(abs_dataset_filename)
            self.raw_dataset[dataset_filename] = raw_dataset_content