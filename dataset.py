import os
import config
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

    def get_raw_batch(self, size, offset, feature_ids, training, param=None):
        batch = []
        for i in range(size):
            raw_features = self.gather_raw_features(i + offset, feature_ids, training, param)
            batch.append(raw_features)
        return batch

    def gather_raw_features(self, index, feature_ids, training, param):
        res = {}
        for feature_id in feature_ids:
            if feature_id in config.index_features:
                res[feature_id] = self.get_raw_feature_by_index(index, feature_id, training, param)
            elif feature_id in config.id_features:
                customer_id = self.get_customer_id(training, index)
                res[feature_id] = self.get_raw_feature_by_id(customer_id, feature_id, param)
            else:
                raise RuntimeError('unknown feature id')
        return res
    
    def get_raw_feature_by_index(self, index, feature_id, training, param):
        if feature_id == 'feature_1':
            return self.get_feature_1(index, training, param)
        elif feature_id == 'feature_2':
            return self.get_feature_2(index, training, param)
        elif feature_id == 'feature_3':
            return self.get_feature_3(index, training, param)
        else:
            raise RuntimeError('unknown feature ID')

    def get_raw_feature_by_id(self, customer_id, feature_id, param):
        return 0
    
    def get_customer_id(self, training, index):
        if training:
            return self.raw_dataset['train']['customer_id'][index]
        else:
            return self.raw_dataset['test']['customer_id'][index]

    def get_feature_1(self, index, training, param):
        if training:
            return self.raw_dataset['train']['feature_1'][index]
        else:
            return self.raw_dataset['test']['feature_1'][index]

    def get_feature_2(self, index, training, param):
        if training:
            return self.raw_dataset['train']['feature_2'][index]
        else:
            return self.raw_dataset['test']['feature_2'][index]

    def get_feature_3(self, index, training, param):
        if training:
            return self.raw_dataset['train']['feature_3'][index]
        else:
            return self.raw_dataset['test']['feature_3'][index]
