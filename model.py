from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers import Dense
from keras.optimizers import SGD
from keras.models import load_model
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import config
import stats


def generate_model(model_type, viz, output_dir):
    if model_type == 'sanity_check':
        return SanityCheckModel(viz, output_dir)
    else:
        raise RuntimeError('unknown model type')

class SanityCheckModel():
    def __init__(self, viz, output_dir):
        self.feature_ids = ['feature_1', 'feature_2', 'feature_3', 'card_id', 'first_active_month', 'new_all']
        self.target_ids = ['target']
        self.kmodel = None
        self.viz = viz
        self.output_dir = output_dir
    
    def get_feature_ids(self):
        return self.feature_ids
    
    def get_target_ids(self):
        return self.target_ids + self.feature_ids

    def preprocess_raw_feature(self, raw_feature, training):
        feature = {'x': None, 'y': None}
        feature_1 = [0,0,0,0,0]
        feature_1[raw_feature['feature_1'] - 1] = 1
        feature_2 = [0,0,0]
        feature_2[raw_feature['feature_2'] - 1] = 1
        feature_3 = [raw_feature['feature_3']]
        first_active_month = self.convert_first_active_month(raw_feature['first_active_month'])
        raw_new_all = raw_feature['new_all']
        state_feature = self.count_state_id(raw_new_all)
        feature['x'] = feature_1 + feature_2 + feature_3 + first_active_month + state_feature
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

    def init_model(self, load_trained_model, config=None):
        if load_trained_model:
            model_path = os.path.join(self.output_dir, 'model.h5')
            print('loading model from ' + model_path)
            self.kmodel = load_model(model_path)
        else:
            print('constructing model ...')
            self.compose_model(config)
        print('model initialized')
    
    def compose_model(self, config):
        kmodel = Sequential()
        kmodel.add(Dense(units=64, activation='relu', input_dim=35))
        kmodel.add(Dense(units=64, activation='relu'))
        kmodel.add(BatchNormalization())
        kmodel.add(Dense(units=32, activation='relu'))
        kmodel.add(BatchNormalization())
        kmodel.add(Dense(units=1, activation='tanh'))
        kmodel.compile(loss='mean_squared_error', optimizer='sgd')
        self.kmodel = kmodel

    def train(self, raw_fearure_batch, epoch):
        batch_feature = self.preprocess_batch(raw_fearure_batch, True)
        loss = self.kmodel.fit(batch_feature['x'], batch_feature['y'], verbose=0, epochs=epoch, validation_split=0.15)
        if self.viz:
            plt.figure()
            plt.plot(loss.history['loss'])
            plt.plot(loss.history['val_loss'])
            plt.show()
        model_save_path = os.path.join(self.output_dir, 'model.h5')
        self.kmodel.save(model_save_path)

    def validate(self, raw_fearure_batch):
        batch_feature = self.preprocess_batch(raw_fearure_batch, True)
        res = self.kmodel.evaluate(batch_feature['x'], batch_feature['y'], verbose=0)
        return res
    
    def test(self, raw_fearure_batch):
        batch_feature = self.preprocess_batch(raw_fearure_batch, False)
        predictions = self.kmodel.predict(batch_feature['x'], verbose=0)
        flat_predictions = [pred[0] for pred in predictions]
        denormalized_predictions = self.denormalize_target(flat_predictions)
        card_ids = [feature['card_id'] for feature in raw_fearure_batch]
        res = {'raw': flat_predictions, 'denormalized': denormalized_predictions, 'card_id': card_ids}
        return res

    def normalize_target(self, target):
        return [np.tanh(target)]
        
    def denormalize_target(self, targets):
        res = []
        for target in targets:
            if target <= -1.0:
                res.append(stats.target_upperbound)
            elif target >= 1.0:
                res.append(stats.target_lowerbound)
            else:
                denormalized_target = np.arctanh(target)
                res.append(denormalized_target)
        return res
    
    def convert_first_active_month(self, date_str):
        res = []
        try:
            clean_data = date_str.strip()
            current_time = datetime.strptime(clean_data, config.date_format).timestamp()
            normalized_time = self.normalize_first_active_month(current_time)
            res = [normalized_time]
        except:
            res = [0.5]
        return res
    
    def normalize_first_active_month(self, timestamp):
        normalized_time = (timestamp - stats.min_timestamp) / stats.timespan
        return normalized_time

    def count_state_id(self, raw_new_all):
        state_count = [0.0 for i in range(stats.state_id_count)]
        if len(raw_new_all) == 0:
            return state_count
        trans_count = float(len(raw_new_all))
        for index, row in raw_new_all.iterrows():
            int_state_id = int(row['state_id'])
            index = int_state_id if int_state_id != -1 else 0
            state_count[index] = state_count[index] + 1.0
        normalized_state_feature = [cnt / trans_count for cnt in state_count]
        return normalized_state_feature

