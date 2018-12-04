import config
import dataset
import model


def common_routine(dataset_root, validation_size, batch_size, train_iter, viz=False):
    dataset_object = dataset.Dataset()
    experiment_model = model.generate_model('sanity_check', viz)
    experiment_model.init_model()
    dataset_object.load_raw_dataset(dataset_root, config.dataset_meta, 'train')
    dataset_object.load_raw_dataset(dataset_root, config.dataset_meta, 'test')
    dataset_object.load_raw_dataset(dataset_root, config.dataset_meta, 'new_merchant_transactions')
    train_batch_size = dataset_object.get_size(True) - validation_size
    count = 0
    validate_data = dataset_object.get_raw_batch(validation_size, train_batch_size, experiment_model.get_target_ids(), training=True)
    test_batch = dataset_object.get_raw_batch(dataset_object.get_size(False), 0, experiment_model.get_feature_ids(), training=False)
    while count < train_batch_size:
        print('starting ' + str(count) + ' out of ' + str(train_batch_size) + ', finishing ' + str(count/train_batch_size*100) + '% ...')
        size = min(batch_size, train_batch_size - count)
        train_batch = dataset_object.get_raw_batch(size, count, experiment_model.get_target_ids(), training=True)
        experiment_model.train(train_batch, train_iter)
        train_err = experiment_model.validate(train_batch)
        print('training error: ' + str(train_err))
        test_err = experiment_model.validate(validate_data)
        print('testing error: ' + str(test_err))
        experiment_model.test(test_batch, dataset_root)
        count = count + batch_size
    print('all finished!')
