from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
import numpy as np
import pandas as pd
import os
import stats


def generate_model(model_type):
    if model_type == 'sanity_check':
        return SanityCheckModel()
    else:
        raise RuntimeError('unknown model type')

class SanityCheckModel():
    def __init__(self):
        self.feature_ids = ['feature_1', 'feature_2', 'feature_3', 'card_id']
        self.target_ids = ['target']
        self.kmodel = None
    
    def get_feature_ids(self):
        return self.feature_ids
    
    def get_target_ids(self):
        return self.target_ids + self.feature_ids

    def preprocess_raw_feature(self, raw_feature, training):
        feature = {'x': None, 'y': None}
        feature_1 = [0,0,0,0,0]
        feature_1[raw_feature['feature_1']-1] = 1
        feature_2 = [0,0,0]
        feature_2[raw_feature['feature_2']-1] = 1
        feature_3 = [raw_feature['feature_3']]
        feature['x'] = feature_1 + feature_2 + feature_3
        if training:
            target = raw_feature['target']
            feature['y'] = self.normalize_target(target)
        return feature
    
    def preprocess_batch(self, raw_feature_batch, training):
        batch_feature_x = []
        batch_feature_y = []
        for raw_feature in raw_feature_batch:
            processed_feature = self.preprocess_raw_feature(raw_feature, training)
            batch_feature_x.append(processed_feature['x'])
            if training:
                batch_feature_y.append(processed_feature['y'])
        batch_feature = {'x': np.array(batch_feature_x), 'y': np.array(batch_feature_y)}
        return batch_feature

    def init_model(self, config=None):
        kmodel = Sequential()
        kmodel.add(Dense(units=64, activation='relu', input_dim=9))
        kmodel.add(Dense(units=128, activation='relu'))
        kmodel.add(Dense(units=1, activation='linear'))
        sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
        kmodel.compile(loss='mean_squared_error', optimizer=sgd)
        self.kmodel = kmodel

    def train(self, raw_fearure_batch, epoch):
        batch_feature = self.preprocess_batch(raw_fearure_batch, True)
        loss = self.kmodel.fit(batch_feature['x'], batch_feature['y'], verbose=1, epochs=epoch)

    def validate(self, raw_fearure_batch):
        batch_feature = self.preprocess_batch(raw_fearure_batch, True)
        self.kmodel.evaluate(batch_feature['x'], batch_feature['y'])
    
    def test(self, raw_fearure_batch, output_dir):
        batch_feature = self.preprocess_batch(raw_fearure_batch, False)
        predictions = self.kmodel.predict(batch_feature['x'], verbose=1)
        flat_predictions = [pred[0] for pred in predictions]
        denormalized_predictions = self.denormalize_target(flat_predictions)
        card_ids = [feature['card_id'] for feature in raw_fearure_batch]
        df = pd.DataFrame({'card_id': card_ids, 'target': flat_predictions})
        output_path = os.path.join(output_dir, 'submission.csv')
        df.to_csv(output_path, index=False)

    def normalize_target(self, target):
        normalized_target = 0.0
        if target < stats.target_lowerbound:
            normalized_target = 0.0
        elif target > stats.target_upperbound:
            normalized_target = 1.0
        else:
            normalized_target = (target + stats.target_shift) / stats.target_scale
        return [normalized_target]
        
    def denormalize_target(self, targets):
        res = []
        for target in targets:
            denormalized_target = target * stats.target_scale - stats.target_shift
            res.append(denormalized_target)
        return res
        
