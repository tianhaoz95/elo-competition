from keras.models import Sequential
from keras.layers import Dense
import numpy as np


def generate_model(model_type):
    if model_type == 'sanity_check':
        return SanityCheckModel()
    else:
        raise RuntimeError('unknown model type')

class SanityCheckModel():
    def __init__(self):
        self.feature_ids = ['feature_1', 'feature_2', 'feature_3']
        self.kmodel = None
    
    def get_feature_ids(self):
        return self.feature_ids

    def preprocess_raw_feature(self, raw_feature):
        feature = {}
        feature_1 = raw_feature['feature_1']
        feature_2 = raw_feature['feature_2']
        feature_3 = raw_feature['feature_3']
        target = raw_feature['target']
        feature['x'] = [feature_1, feature_2, feature_3]
        feature['y'] = [target]
        return feature
    
    def preprocess_batch(self, raw_feature_batch):
        batch_feature_x = []
        batch_feature_y = []
        for raw_feature in raw_feature_batch:
            processed_feature = self.preprocess_raw_feature(raw_feature)
            batch_feature_x.append(processed_feature['x'])
            batch_feature_y.append(processed_feature['y'])
        batch_feature = {'x': np.array(batch_feature_x), 'y': np.array(batch_feature_y)}
        return batch_feature

    def init_model(self, config=None):
        kmodel = Sequential()
        kmodel.add(Dense(units=64, activation='relu', input_dim=3))
        kmodel.add(Dense(units=128, activation='relu'))
        kmodel.add(Dense(units=1, activation='linear'))
        kmodel.compile(loss='mean_squared_error', optimizer='sgd')
        self.kmodel = kmodel

    def train(self, raw_fearure_batch):
        batch_feature = self.preprocess_batch(raw_fearure_batch)
        self.kmodel.train_on_batch(batch_feature['x'], batch_feature['y'])

    def validate(self):
        pass
    
    def test(self):
        pass