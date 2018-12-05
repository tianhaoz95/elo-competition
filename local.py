import util

def main():
    print('Start local sanity testing ...')
    dataset_root = './data'
    util.common_routine(dataset_root, 10, 10, 10, viz=False, load_trained_model=True)

if __name__ == '__main__':
    main()
