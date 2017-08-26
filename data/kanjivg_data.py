"""
data loader for KanjiVG dataset

"""


import numpy as np
import pickle
from os import path
from pathlib import Path
import random
from .item2vec import train_item2vec

import settings

DATASET_PATH = path.join(settings.DATA_STORE_PATH, 'kanjivg', 'data')

TEST_NUMBER=100


class DataLoader(object):
    """ an object that generates batches of MovieLens_100K data for training """

    def __init__(self, args):
        """
        Initialize the DataLoader
        :param args: all kinds of argument
        """

        self.data_dir = DATASET_PATH

        if not Path(DATASET_PATH).is_file():
            print("The data file for kanjivg doesn't exist")
            quit()

        # load ML_100K data to RAM

        print('user number',self.user_num)

        # hist_collect = self.get_item_matrix()
        # mask = [x for x in range(self.user_num)]
        #
        # random.Random(args.seed).shuffle(mask)
        #
        # train_num = int(round(self.user_num - TEST_NUMBER))
        #
        # train_hist = [hist_collect[it] for it in mask[:train_num]]  # type: list[list]
        # val_hist = [hist_collect[it] for it in mask[train_num:]]   # type: list[list]

        self.max_len = args.hist_length

        if not len(self.X_train) == len(self.y_train):
            print("The number of records in X_data and Y_data is not equal!")
            quit()
        print("The number of training records is", len(self.X_train))
        self.X_train = np.asarray(self.X_train)
        self.y_train = np.asarray(self.y_train)

        self.X_val = []
        self.y_val = []
        pass
