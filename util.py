import config
import dataset


def common_routine(dataset_root):
    dataset_object = dataset.Dataset()
    dataset_object.load_raw_dataset(dataset_root, config.dataset_meta, 'train')
    dataset_object.show_raw_brief(config.dataset_meta, 'train')
    dataset_object.load_raw_dataset(dataset_root, config.dataset_meta, 'test')
    dataset_object.show_raw_brief(config.dataset_meta, 'test')