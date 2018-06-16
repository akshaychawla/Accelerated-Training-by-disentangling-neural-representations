"""Data generators. Lots of them. Or some of them. I don't care."""

from __future__ import print_function
from __future__ import division

import sys
from keras.datasets import cifar10
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
from sklearn.utils import shuffle
import numpy as np


class dg_cifar10:
    def __init__(self, batch_size, embedding_units=None, mode=None):
        self.mode = mode
        self.batch_size = batch_size
        self.embedding_units = embedding_units
        self.num_triplets = self.batch_size // 3

        ## loading cifar10 data
        (self.x_train, self.y_train), (self.x_test, self.y_test) = cifar10.load_data()
        (self.x_train, self.y_train) = shuffle(self.x_train, self.y_train)
        (self.x_test, self.y_test) = shuffle(self.x_test, self.y_test)
        self.data_size = self.x_train.shape[0]
        self.test_size = self.x_test.shape[0]
        print("\ntraining data size:", self.data_size)
        print("testing data size:", self.test_size, "\n")

        ## change test data to be divisible by 129
        self.x_test = np.vstack((self.x_test, self.x_test[:62]))
        self.y_test = np.vstack((self.y_test, self.y_test[:62]))

        self.y_test = to_categorical(self.y_test, num_classes=10)

        ## create the Keras ImageDataGenerator (Train+Test)
        self.train_dgen = ImageDataGenerator(
                        featurewise_center=True,
                        featurewise_std_normalization=True,
                        horizontal_flip=True,
                        width_shift_range=4, # mimick padding=4 + randomcrop
                        height_shift_range=4,
                        fill_mode="nearest"
                    )
        self.train_dgen.fit(self.x_train)
        self.test_dgen = ImageDataGenerator(
                        featurewise_center=True,
                        featurewise_std_normalization=True
                    )
        self.test_dgen.fit(self.x_train)

        if mode == "triplet":
            if embedding_units is None:
                print("ERROR. Supplied triplet mode but not embedding_units??")
                sys.exit()

            if self.batch_size%3 != 0:
                print("ERROR. Supplied triplet mode but batch_size is not divisible by 3.")
                sys.exit()

        elif mode == "normal":
            if embedding_units is not None:
                print("ERROR. Supplied normal mode but also embedding_units??")
                sys.exit()

        else:
            print("ERROR. Unknown mode supplied.")
            sys.exit()


    def TRAIN_single_triplet_generator(self):
        """
        Generates a single training pair for both loss functions.
        """
        while True:
            anc_class = np.random.randint(0, 10)
            neg_class = None
            while (neg_class is None) or (neg_class == anc_class):
                neg_class = np.random.randint(0, 10)
            # checks
            assert neg_class != anc_class

            idx_anchors = np.argwhere(self.y_train==anc_class)[:, 0]
            anchor_xs = self.x_train[idx_anchors]
            anchor_ys = self.y_train[idx_anchors]
            _args = np.random.choice(anchor_xs.shape[0], size=2, replace=False)
            anc, pos = anchor_xs[_args]
            y_anc, y_pos = to_categorical(anchor_ys[_args], num_classes=10)

            idx_negs = np.argwhere(self.y_train==neg_class)[:, 0]
            neg_xs = self.x_train[idx_negs]
            neg_ys = self.y_train[idx_negs]
            _narg = np.random.choice(neg_xs.shape[0], size=1)[0]
            neg = neg_xs[_narg]
            y_neg = to_categorical(neg_ys[_narg], num_classes=10)[0]

            yield (anc, pos, neg), (y_anc, y_pos, y_neg)


    def TRAIN_batched_triplet_generator(self):
        """
        Generates a single batch for both loss functions.
        Then passes the data through the ImageDataGenerator.
        Because Keras.
        """
        tgen = self.TRAIN_single_triplet_generator()
        while True:
            L_anc, L_pos, L_neg = [], [], []
            Y_anc, Y_pos, Y_neg = [], [], []

            for _ in range(self.num_triplets):
                (anc, pos, neg), (y_anc, y_pos, y_neg) = next(tgen)
                L_anc.append(anc)
                L_pos.append(pos)
                L_neg.append(neg)

                Y_anc.append(y_anc)
                Y_pos.append(y_pos)
                Y_neg.append(y_neg)

            batch = np.vstack((L_anc, L_pos, L_neg))
            truth = np.vstack((Y_anc, Y_pos, Y_neg))

            batch = self.train_dgen.flow(batch, batch_size=self.batch_size, shuffle=False).next()

            yield  batch, {"final_norms":np.zeros((self.batch_size, self.embedding_units)),
                           "preds":truth}

    def TEST_batched_triplet_generator(self, test_bs=100):
        """
        Only need to evaluate on preds.
        """
        self.test_bs = test_bs
        dummy_norms = np.zeros((test_bs, self.embedding_units))

        flowing_data = self.test_dgen.flow(self.x_test, self.y_test, batch_size=test_bs, shuffle=False)

        while True:
            batch, truth = flowing_data.next()
            print(flowing_data.total_batches_seen, batch.shape, truth.shape)
            if batch.shape[0] == test_bs:
                yield batch, {"final_norms":dummy_norms, "preds":truth}
            else:
                yield batch, {"final_norms":np.zeros((batch.shape[0], self.embedding_units)), "preds":truth}

def test_data_generators():
    """
    Utility to test train + test data generators
    """

    # Test train data gen
    dg = dg_cifar10(129, 300, "triplet")
    TRAIN_dgen = dg.TRAIN_batched_triplet_generator()
    x,gt = next(TRAIN_dgen)
    assert x.shape == (129,32,32,3)
    assert gt["preds"].shape == (129,10)
    assert gt["final_norms"].shape == (129,300)

    # Test TEST data gen
    TEST_dgen = dg.TEST_batched_triplet_generator(test_bs=50)
    x,gt = next(TEST_dgen)
    assert x.shape == (50,32,32,3)
    assert gt["preds"].shape == (50,10)
    assert gt["final_norms"].shape == (50,300)

    # Check speed
    import time
    time_per_call = []
    start_time = time.time()
    for _ in range(30):
        data_tuple = next(TEST_dgen)
        stop_time = time.time()
        time_per_call.append(stop_time - start_time)
        start_time = stop_time
    print("Max:{:.3f} | Min:{:.3f} | Mean:{:.3f} |".format(
            max(time_per_call), min(time_per_call),
            sum(time_per_call) / len(time_per_call))
        )

    print("..Basic tests passed.")


if __name__ == '__main__':
    test_data_generators()
    # dg = dg_cifar10(129, 300, "triplet")
    # triplet_generator = dg.TRAIN_batched_triplet_generator()
    # import ipdb; ipdb.set_trace()
    # data = triplet_generator.next()

    print("YEAH")
