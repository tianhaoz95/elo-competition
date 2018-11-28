import config
import dataset


def load_dataset(dataset_root):
    res = dataset.Dataset()
    res.load_raw_dataset(dataset_root, config.dataset_meta, 'train')
    res.show_raw_brief(config.dataset_meta, 'train')
    res.load_raw_dataset(dataset_root, config.dataset_meta, 'test')
    res.show_raw_brief(config.dataset_meta, 'test')
    return res
