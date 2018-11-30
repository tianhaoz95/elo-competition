import config
import dataset
import model


def common_routine(dataset_root):
    dataset_object = dataset.Dataset()
    experiment_model = model.generate_model('sanity_check')
    dataset_object.load_raw_dataset(dataset_root, config.dataset_meta, 'train')
    dataset_object.show_raw_brief(config.dataset_meta, 'train')
    dataset_object.load_raw_dataset(dataset_root, config.dataset_meta, 'test')
    dataset_object.show_raw_brief(config.dataset_meta, 'test')
    batch = dataset_object.get_raw_batch(10, 0, ['feature_1', 'feature_2', 'feature_3', 'target'], training=True)
    experiment_model.init_model()
    experiment_model.train(batch)