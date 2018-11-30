from keras.models import Sequential
from keras.layers import Dense


class SanityCheckModel():
    def __init__(self):
        self.feature_ids = ['feature_1', 'feature_2', 'feature_3']
        self.kmodel = None
    
    def get_feature_ids(self):
        return self.feature_ids

    def preprocess_raw_feature(self):
        pass
    
    def preprocess_batch(self, raw_feature_batch):
        batch_feature = []
        for raw_feature in raw_feature_batch:
            processed_feature = self.preprocess_raw_feature(raw_feature)
            batch_feature.append(processed_feature)
        return batch_feature

    def init_model(self, config=None):
        kmodel = Sequential()
        kmodel.add(Dense(units=64, activation='relu', input_dim=100))
        kmodel.add(Dense(units=128, activation='relu'))
        kmodel.add(Dense(units=10, activation='softmax'))
        kmodel.compile(loss='mean_squared_error', optimizer='sgd')
        self.kmodel = kmodel

    def train(self, raw_fearure_batch):
        batch_feature = self.preprocess_batch(raw_fearure_batch)
        self.kmodel.train_on_batch(batch_feature['x'], batch_feature['y'])

    def validate(self):
        pass
    
    def test(self):
        pass