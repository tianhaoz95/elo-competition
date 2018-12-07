import config
import dataset
import model
import os
import pandas as pd
import numpy as np


def common_routine(dataset_root, validation_size, batch_size, train_iter, viz=False, output_dir='./', test=True, train=True, load_trained_model=False, train_limit=None, index_type='random'):
    dataset_object = dataset.Dataset()
    experiment_model = model.generate_model('sanity_check', viz, output_dir)
    experiment_model.init_model(load_trained_model)
    dataset_object.load_raw_dataset(dataset_root, config.dataset_meta, 'train')
    dataset_object.load_raw_dataset(dataset_root, config.dataset_meta, 'test')
    dataset_object.load_raw_dataset(dataset_root, config.dataset_meta, 'new_merchant_transactions')
    print('dataset files loaded')
    if train:
        if index_type == 'range':
            train_range_routine(dataset_object, experiment_model, batch_size, output_dir, validation_size, train_iter, train_limit, index_type)
        if index_type == 'random':
            train_random_routine(dataset_object, experiment_model, batch_size, output_dir, validation_size, train_iter)
    if test:
        test_routine(dataset_object, experiment_model, batch_size, output_dir)
    print('all finished!')

def train_random_routine(dataset_object, experiment_model, batch_size, output_dir, validation_size, train_iter):
    train_batch_size = dataset_object.get_size(training=True) - validation_size
    validate_data = dataset_object.get_raw_batch_from_range(validation_size, train_batch_size, experiment_model.get_target_ids(), training=True)
    print('start random training ...')
    for i in range(train_iter):
        print('starting ' + str(i) + ' out of ' + str(train_iter) + ', finishing ' + str(i/train_iter*100) + '% ...')
        train_batch = dataset_object.get_raw_batch_from_indexes(np.random.randint(low=0, high=train_batch_size, size=batch_size), experiment_model.get_target_ids(), training=True)
        experiment_model.train(train_batch, epochs=train_iter, validate_feature_batch=validate_data)
        train_err = experiment_model.validate(train_batch)
        print('training error: ' + str(train_err))
        test_err = experiment_model.validate(validate_data)
        print('testing error: ' + str(test_err))
    print('random training done')

def train_range_routine(dataset_object, experiment_model, batch_size, output_dir, validation_size, train_iter, train_limit):
    train_batch_size = dataset_object.get_size(training=True) - validation_size
    validate_data = dataset_object.get_raw_batch_from_range(validation_size, train_batch_size, experiment_model.get_target_ids(), training=True)
    print('start range training ...')
    count = 0
    actual_train_batch_size = train_batch_size
    if train_limit is not None:
        actual_train_batch_size = min(train_limit, train_batch_size)
    while count < actual_train_batch_size:
        print('starting ' + str(count) + ' out of ' + str(actual_train_batch_size) + ', finishing ' + str(count/actual_train_batch_size*100) + '% ...')
        size = min(batch_size, actual_train_batch_size - count)
        train_batch = dataset_object.get_raw_batch_from_range(size, count, experiment_model.get_target_ids(), training=True)
        experiment_model.train(train_batch, epochs=train_iter, validate_feature_batch=validate_data)
        train_err = experiment_model.validate(train_batch)
        print('training error: ' + str(train_err))
        test_err = experiment_model.validate(validate_data)
        print('testing error: ' + str(test_err))
        count = count + batch_size
    print('range training done')

def test_routine(dataset_object, experiment_model, batch_size, output_dir):
    print('start testing ...')
    test_size = dataset_object.get_size(training=False)
    count = 0
    raw_predictions = []
    card_ids = []
    denormalized_predictions = []
    while count < test_size:
        print('starting ' + str(count) + ' out of ' + str(test_size) + ', finishing ' + str(count/test_size*100) + '% ...')
        size = min(batch_size, test_size - count)
        test_batch = dataset_object.get_raw_batch_from_range(size, count, experiment_model.get_feature_ids(), training=False)
        predictions = experiment_model.test(test_batch)
        raw_predictions = raw_predictions + predictions['raw']
        card_ids = card_ids + predictions['card_id']
        denormalized_predictions = denormalized_predictions + predictions['denormalized']
        count = count + batch_size
    raw_res_df = pd.DataFrame({'card_id': card_ids, 'target': raw_predictions})
    res_df = pd.DataFrame({'card_id': card_ids, 'target': denormalized_predictions})
    raw_output_path = os.path.join(output_dir, 'raw_submission.csv')
    output_path = os.path.join(output_dir, 'submission.csv')
    raw_res_df.to_csv(raw_output_path, index=False)
    res_df.to_csv(output_path, index=False)
    print('test done')
