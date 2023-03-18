import sys
import itertools
import numpy as np
from config import *

# def show_images(images, labels, index):
#     import matplotlib.pyplot as plt
#     label_names = get_label_names()
#     fig = plt.figure()
#     for i in range(48):
#         ax = fig.add_subplot(6, 8, i + 1)
#         ax.imshow(images[index + i].reshape(3, 32, 32).transpose(1, 2, 0))
#         # ax.set_title(labels[index + i])
#         ax.set_title(label_names[labels[index + i]])
#         ax.set_xticks([])
#         ax.set_yticks([])
#     plt.show()


def choose_data_train(data, stu_id):
    label_index = list(range(NUM_LABEL_BEFORE))
    index_all = tuple(itertools.permutations(label_index,
                                             NUM_LABEL_AFTER))[1:NUM_BATCH + 1]
    index_choose = index_all[stu_id % NUM_BATCH]
    # iter data, remove the data with label not in index_choose
    data_iter = []
    for i in range(data.shape[0]):
        if data[i][-1] in index_choose:
            data_iter.append(data[i])
    data_iter = np.array(data_iter)
    return data_iter


def choose_data_test_x(data_x, stu_id):
    index_test_all = np.load(INDEX_TEST_ALL)
    index_test_choosed = index_test_all[stu_id % NUM_BATCH]
    data_x_choosed = data_x[index_test_choosed]
    return data_x_choosed


def chose_data_test_y(data_y, stu_id):
    index_test_all = np.load(INDEX_TEST_ALL)
    index_test_choosed = index_test_all[stu_id % NUM_BATCH]
    data_y_choosed = data_y[index_test_choosed]
    return data_y_choosed


if __name__ == '__main__':
    stu_id = int(sys.argv[1])
    data_train = np.load(DATA_TRAIN)
    data_test_x = np.load(DATA_TEST_X)
    data_train_choosed = choose_data_train(data_train, stu_id)
    data_test_x_choosed = choose_data_test_x(data_test_x, stu_id)
    print(data_train_choosed.shape)
    print(data_test_x_choosed.shape)
    np.save(STU_DATA_TRAIN, data_train_choosed)
    np.save(STU_DATA_TEST_X, data_test_x_choosed)

    # date_test_y = np.load(DATA_TEST_Y)
    # for i in range(NUM_BATCH):
    #     data_test_y_choosed = chose_data_test_y(date_test_y, i)
    #     print(data_test_y_choosed.shape)
    #     print(set(data_test_y_choosed.tolist()))
    #     np.save("label_batch/label_batch_{}".format(i), data_test_y_choosed)

    # for istu_id in [20122111, 20122112, 20122113]:
    #     istu_id = int(istu_id % NUM_BATCH)
    #     data_train = np.load(DATA_TRAIN)
    #     data_test_x = np.load(DATA_TEST_X)
    #     data_test_y = np.load(DATA_TEST_Y)
    #     data_train_choosed = choose_data_train(data_train, istu_id)
    #     data_test_x_choosed = choose_data_test_x(data_test_x, istu_id)
    #     data_test_y_choosed = chose_data_test_y(data_test_y, istu_id)
    #     print(set(data_test_y_choosed.tolist()))
    #     show_images(data_test_x, data_test_y, 0)
