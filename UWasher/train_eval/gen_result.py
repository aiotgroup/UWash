import scipy.io as scio


if __name__ == '__main__':
    ret_mat = scio.loadmat("./normal_for_confusion_matrix.mat")
    print(ret_mat['prediction'].shape)
    test_ranges = scio.loadmat("../data/result_ranges.mat")
    print(test_ranges)
    test_data = scio.loadmat("../data/result_data.mat")
    print(test_data['acc']['canteen_2'][0][0].shape)