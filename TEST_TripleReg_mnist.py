## Extension of TEST_tripletloss_mnist.
## This treats the triplet loss as a regularizer.
## Possible usecases : faster training, regularizer against FGSM.

from __future__ import print_function
from keras.callbacks import TensorBoard, ReduceLROnPlateau, Callback
from keras.layers import Input, Conv2D, MaxPooling2D, Dropout, Flatten, Lambda, Dense
from keras.models import Sequential, Model
import keras.backend as K
from keras import optimizers
from keras.datasets import mnist
from keras.utils import to_categorical
from loss_layers import triplet_loss_batched_wrapper, wrapper_categorical_crossentropy
from keras.losses import categorical_crossentropy
from keras.optimizers import Adadelta

import numpy as np
import time
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.manifold import TSNE

def nw_arch_mnist():
    """
    LeNet architecture + triplet loss
    """
    EMBEDDING_UNITS = 64
    input_shape = (28,28,1)
    num_classes = 10

    inputs = Input(shape=input_shape, name='inputs')
    conv1 = Conv2D(32, kernel_size=(3, 3), activation='relu', name="conv1")(inputs)
    conv2 = Conv2D(64, kernel_size=(3, 3), activation='relu', name="conv2")(conv1)
    pool = MaxPooling2D(pool_size=(2, 2), name="pool")(conv2)
    drop = Dropout(0.25)(pool)
    flat = Flatten()(drop)
    dense = Dense(128, activation='relu')(flat)
    drop2 = Dropout(0.5)(dense)
    ## ...now, BRANCH!!!
    embed = Dense(EMBEDDING_UNITS, name='embeds')(drop2) ## embeddings for aux loss
    norms = Lambda(lambda x: K.l2_normalize(x, axis=-1), name="norms")(embed) ## normed embeddings
    preds = Dense(num_classes, activation='softmax', name='preds')(drop2) ## standard loss

    model = Model(inputs=inputs, outputs=[norms, preds])
    return model

"""
PREP DATA
"""
EMBEDDING_UNITS = 64

def triplet_generator():
    """
    Returns single data for both loss functions.
    """
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, y_train = shuffle(x_train, y_train)

    while True:
        anc_class = np.random.randint(0, 10)
        neg_class = None
        while (neg_class is None) or (neg_class == anc_class):
            neg_class = np.random.randint(0, 10)
        # checks
        assert neg_class != anc_class

        anchor_xs = x_train[y_train==anc_class]
        anchor_ys = y_train[y_train==anc_class]
        _args = np.random.choice(anchor_xs.shape[0], size=2, replace=False)
        anc, pos = anchor_xs[_args]
        y_anc, y_pos = to_categorical(anchor_ys[_args], num_classes=10)

        neg_xs = x_train[y_train==neg_class]
        neg_ys = y_train[y_train==neg_class]
        _narg = np.random.choice(neg_xs.shape[0], size=1)[0]
        neg = neg_xs[_narg]
        y_neg = to_categorical(neg_ys[_narg], num_classes=10)

        yield (anc, pos, neg), (y_anc, y_pos, y_neg)

def batched_data_generator(batch_size):
    """
    Generates data for both loss functions.
    """
    tgen = triplet_generator()
    while True:
        L_anc, L_pos, L_neg = [], [], []
        Y_anc, Y_pos, Y_neg = [], [], []

        for _ in range(batch_size):
            (anc, pos, neg), (y_anc, y_pos, y_neg) = next(tgen)
            L_anc.append(anc)
            L_pos.append(pos)
            L_neg.append(neg)

            Y_anc.append(y_anc)
            Y_pos.append(y_pos)
            Y_neg.append(y_neg)

        batch = np.vstack((L_anc, L_pos, L_neg))
        batch = np.expand_dims(batch, axis=3)
        truth = np.vstack((Y_anc, Y_pos, Y_neg))

        yield  batch, {"norms":np.zeros((batch_size*3,EMBEDDING_UNITS)), "preds":truth}

def _test_generator():
    dgen = batched_data_generator(batch_size=32)
    data, _ = next(dgen)
    target = _.get("preds")

    print("\n\t--testing data generator--")
    for idx in range(8):
        anc_idx = np.random.randint(0, 32)
        catted  = np.hstack([data[anc_idx,:,:,0],data[anc_idx+32,:,:,0], data[anc_idx+32+32,:,:,0]])
        print(np.argmax(target[[anc_idx, anc_idx+32, anc_idx+32+32]], axis=1))
        plt.subplot(8,1,idx+1)
        plt.imshow(catted)
    plt.show()
    print("\t--done--\n")

_test_generator()

"""
Validation method
"""
class valid_callback(Callback):

    def __init__(self, top_k, num_samples):
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        x_test, y_test = shuffle(x_test, y_test)
        self.x_test = x_test
        self.y_test = y_test
        self.top_k = top_k
        self.num_samples = num_samples

    def on_epoch_end(self, epoch, logs):
        allEmbeds = self.model.predict(np.expand_dims(self.x_test, axis=3), batch_size=256)
        print("computed embeddings for test set")
        commons = []
        for _ in range(self.num_samples):

            valIdx = np.random.randint(0, len(self.x_test))
            valclass = self.y_test[valIdx]
            valembed = allEmbeds[valIdx]

            diffs    = np.linalg.norm(allEmbeds - valembed, axis=1)
            top_k_indices_Embeds = np.argsort(diffs)[:self.top_k]

            # How many of the top K closest samples to the valembed belong to the same class as valembed
            top_k_classes = self.y_test[top_k_indices_Embeds]
            commons.append(np.count_nonzero(top_k_classes==valclass))

        print("Mean Common samples: ", np.mean(commons))


"""
TRAIN MODEL
"""

model = nw_arch_mnist()
loss_triplet = triplet_loss_batched_wrapper(num_triplets=16)
loss_xentropy = wrapper_categorical_crossentropy(num_triplets=16)

model.compile(
    optimizer = Adadelta(),
    loss = {"norms" : loss_triplet, "preds" : categorical_crossentropy},
    loss_weights = {"norms" : 0.0, "preds" : 1.0}
)

lrreduce = ReduceLROnPlateau(monitor="loss", factor=0.1, patience=4, verbose=1, min_lr=1e-08)

dgen = batched_data_generator(batch_size=16)

import ipdb; ipdb.set_trace()

history = model.fit_generator(
    dgen,
    steps_per_epoch=500,
    epochs=50
)

# perform TSNE
(x_train, y_train), _ = mnist.load_data()
embeddings_128d = model.predict(np.expand_dims(self.x_train, axis=3), batch_size=256)
embeddings_2d = TSNE(n_components=2).fit_transform(embeddings_128d)
for label in range(10):
    print("Plotting for label ",label)
    subset_x = embeddings_2d[y_train==label]
    plt.plot(subset_x[:,0], subset[:,1])
plt.show()
