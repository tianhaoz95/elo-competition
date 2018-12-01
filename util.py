import config
import dataset
import model


def common_routine(dataset_root):
    dataset_object = dataset.Dataset()
    experiment_model = model.generate_model('sanity_check')
    experiment_model.init_model()
    training_features = experiment_model.get_target_ids()
    testing_features = experiment_model.get_feature_ids()
    dataset_object.load_raw_dataset(dataset_root, config.dataset_meta, 'train')
    dataset_object.show_raw_brief(config.dataset_meta, 'train')
    dataset_object.load_raw_dataset(dataset_root, config.dataset_meta, 'test')
    dataset_object.show_raw_brief(config.dataset_meta, 'test')
    train_batch = dataset_object.get_raw_batch(10, 0, ['feature_1', 'feature_2', 'feature_3', 'target'], training=True)
    experiment_model.train(train_batch, 10)
    experiment_model.validate(train_batch)
    test_batch = dataset_object.get_raw_batch(10, 0, ['feature_1', 'feature_2', 'feature_3'], training=False)
    experiment_model.test(test_batch)
