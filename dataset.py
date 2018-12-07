import os
import config
import pandas as pd
import numpy as np


class Dataset():
    def __init__(self):
        self.raw_dataset = {}

    def load_raw_dataset(self, dataset_root, dataset_meta, dataset_id):
        dataset_filename = dataset_meta[dataset_id]['filename']
        abs_dataset_filename = os.path.join(dataset_root, dataset_filename)
        print('Loading ' + dataset_id + ' from ' + abs_dataset_filename + '...')
        raw_dataset_content = pd.read_csv(abs_dataset_filename)
        self.raw_dataset[dataset_id] = raw_dataset_content

    def get_size(self, training):
        if training:
            return len(self.raw_dataset['train'])
        else:
            return len(self.raw_dataset['test'])

    def show_raw_brief(self, dataset_meta, dataset_id):
        print(self.raw_dataset[dataset_id].head())
        dataset_size = len(self.raw_dataset[dataset_id])
        print(dataset_id + ' dataset size: ' + str(dataset_size))

    def get_raw_batch_from_range(self, size, offset, feature_ids, training, param=None):
        batch = []
        for i in range(size):
            raw_features = self.gather_raw_features(i + offset, feature_ids, training, param)
            batch.append(raw_features)
        return batch

    def get_raw_batch_from_indexes(self, indexes, feature_ids, training=True, param=None):
        batch = []
        for idx in indexes:
            raw_features = self.gather_raw_features(idx, feature_ids, training, param)
            batch.append(raw_features)
        return batch

    def gather_raw_features(self, index, feature_ids, training, param):
        res = {}
        for feature_id in feature_ids:
            if feature_id in config.index_features:
                res[feature_id] = self.get_raw_feature_by_index(index, feature_id, training, param)
            elif feature_id in config.id_features:
                card_id = self.get_customer_id(training, index)
                res[feature_id] = self.get_raw_feature_by_id(card_id, feature_id, param)
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
        elif feature_id == 'card_id':
            return self.get_card_id(index, training, param)
        elif feature_id == 'target':
            return self.get_target(index, training, param)
        elif feature_id == 'first_active_month':
            return self.get_first_active_month(index, training, param)
        else:
            raise RuntimeError('unknown feature id')

    def get_raw_feature_by_id(self, card_id, feature_id, param):
        matches = self.raw_dataset['new_merchant_transactions'].loc[self.raw_dataset['new_merchant_transactions']['card_id'] == card_id]
        return matches
    
    def get_customer_id(self, training, index):
        if training:
            return self.raw_dataset['train']['card_id'][index]
        else:
            return self.raw_dataset['test']['card_id'][index]

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

    def get_first_active_month(self, index, training, param):
        if training:
            return self.raw_dataset['train']['first_active_month'][index]
        else:
            return self.raw_dataset['test']['first_active_month'][index]

    def get_card_id(self, index, training, param):
        if training:
            return self.raw_dataset['train']['card_id'][index]
        else:
            return self.raw_dataset['test']['card_id'][index]

    def get_target(self, index, training, param):
        if training:
            return self.raw_dataset['train']['target'][index]
        else:
            raise RuntimeError('no target in testing dataset')
