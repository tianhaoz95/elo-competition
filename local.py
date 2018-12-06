"""
This file was enhanced with %% to
be used in Jupyter notebook compatible
environments
"""

#%%
import util

def main():
    print('Start local sanity testing ...')
    dataset_root = './data'
    util.common_routine(dataset_root, 17, 13, 13, viz=False, train=True, test=True, load_trained_model=False, train_limit=70)

if __name__ == '__main__':
    main()
