import config
import dataset

def load_dataset(dataset_root):
    res = dataset.Dataset()
    res.load_raw_dataset(dataset_root, config.dataset_meta)
    return res