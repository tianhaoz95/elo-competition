import os
import pandas as pd


class Dataset():
    def __init__(self):
        self.raw_dataset = {}

    def load_raw_dataset(self, dataset_root, dataset_meta, dataset_id):
        dataset_filename = dataset_meta[dataset_id]['filename']
        abs_dataset_filename = os.path.join(dataset_root, dataset_filename)
        print('Loading ' + dataset_id + ' from ' + abs_dataset_filename + '...')
        raw_dataset_content = pd.read_csv(abs_dataset_filename)
        self.raw_dataset[dataset_id] = raw_dataset_content

    def show_raw_brief(self, dataset_meta, dataset_id):
        print(self.raw_dataset[dataset_id].head())
        dataset_size = len(self.raw_dataset[dataset_id])
        print(dataset_id + ' dataset size: ' + str(dataset_size))

    def gather_sample(self):
        pass
