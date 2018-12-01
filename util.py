import config
import dataset
import model


def common_routine(dataset_root, validation_size, batch_size, train_iter):
    dataset_object = dataset.Dataset()
    experiment_model = model.generate_model('sanity_check')
    experiment_model.init_model()
    training_features = experiment_model.get_target_ids()
    testing_features = experiment_model.get_feature_ids()
    dataset_object.load_raw_dataset(dataset_root, config.dataset_meta, 'train')
    dataset_object.show_raw_brief(config.dataset_meta, 'train')
    dataset_object.load_raw_dataset(dataset_root, config.dataset_meta, 'test')
    dataset_object.show_raw_brief(config.dataset_meta, 'test')
    train_batch_size = dataset_object.get_size(True) - validation_size
    count = 0
    while count < train_batch_size:
        print('starting ' + str(count) + ' out of ' + str(train_batch_size))
        size = min(batch_size, train_batch_size - count)
        train_batch = dataset_object.get_raw_batch(size, count, experiment_model.get_target_ids(), training=True)
        experiment_model.train(train_batch, train_iter)
        count = count + batch_size
    experiment_model.validate(train_batch)
    test_batch = dataset_object.get_raw_batch(dataset_object.get_size(False), 0, experiment_model.get_feature_ids(), training=False)
    experiment_model.test(test_batch, dataset_root)
