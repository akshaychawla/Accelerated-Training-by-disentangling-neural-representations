"""This is to train with triplet loss."""

from __future__ import print_function
from __future__ import division

import numpy as np
import matplotlib.pyplot as plt
import os, sys, time, pickle
import wide_residual_network as wrn
from keras.datasets import cifar10
from keras.callbacks import Callback, ModelCheckpoint, LearningRateScheduler, TensorBoard
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD
from loss_layers import triplet_loss_batched_wrapper
from le_data_generators import dg_cifar10

# mode
mode = sys.argv[1].lower()
print("\n\n\t\tTRAINING MODE: %s\n\n"%mode)
if mode == "triplet":
    root_folder = "./TRIP_{}/".format(time.time())
elif mode == "normal":
    rir = raw_input("\n\nWARNING: you are using TRIPLET_cifar10_WRN_28_10.py but are passing 'normal' as the mode.\nAre you sure you want to continue?<y/n>:")
    if rir == 'y':
        pass
    else:
        print("Responded %s. Stopping."%rir)
        sys.exit()
    root_folder = "./RUN_{}/".format(time.time())
else:
    print("\n\n\t\tINCORRECT MODE ARG RECEIVED. EXITING.\n\n")
    sys.exit()

# hyperparameters
batch_size = 129
epochs  = 200
img_rows, img_cols = 32, 32
weight_decay = 0.0005
embedding_units = 300
test_bs = 129

# Logs + checkpoints directory
os.makedirs(root_folder)
os.makedirs(root_folder+"logs")
os.makedirs(root_folder+"checkpoints")
print("Created folders in .. ",root_folder)

# Dataset
(trainX, trainY), (testX, testY) = cifar10.load_data()
trainY = to_categorical(trainY)
testY = to_categorical(testY)

# Callbacks
def lr_scheduler_fxn(epoch):
    if epoch<60:
        return 0.1
    elif epoch<120:
        return 0.02
    elif epoch<160:
        return 0.004
    else:
        return 0.0008

lrschedule = LearningRateScheduler(lr_scheduler_fxn, verbose=1)
tboard     = TensorBoard(log_dir=os.path.join(root_folder, "logs"))
checkpoint = ModelCheckpoint(
                filepath=os.path.join(root_folder, "checkpoints",
                                    "wts_{epoch:02d}-{loss:.2f}.h5"),
                verbose=True,
                save_weights_only=True
            )

# Network
model, loss_list = wrn.create_wide_residual_network(
            (32,32,3),
            nb_classes=10,
            N=4, k=10,
            dropout=0.0,
            mode=mode
        )
print(model.summary())
sgd = SGD(lr=0.1, momentum=0.9, nesterov=True)

# Loss
if mode == "normal":
    model.compile(loss="categorical_crossentropy", optimizer=sgd, metrics=["accuracy"])
else:
    loss_triplet = triplet_loss_batched_wrapper(num_triplets=batch_size//3)
    loss_dict = {"final_norms" : loss_triplet, "preds" : "categorical_crossentropy"}
    loss_weights = {"final_norms" : 1.0, "preds" : 1.0}

    for loss_tensor in loss_list:
        print(loss_tensor.name)
        llname = loss_tensor.name.split('/')[0]
        loss_dict[llname] = loss_triplet
        loss_weights[llname] = 1.0

    import ipdb; ipdb.set_trace()
    model.compile(
        optimizer = sgd,
        loss = loss_dict,
        loss_weights = loss_weights,
        metrics = {"preds":"accuracy"}
    )

# Data generators
c10dg = dg_cifar10(batch_size, embedding_units, "triplet", num_losses=len(loss_list))
train_triplet_generator = c10dg.TRAIN_batched_triplet_generator()
test_triplet_generator = c10dg.TEST_batched_triplet_generator(test_bs)

# Train
history = model.fit_generator(
        generator=train_triplet_generator,
        steps_per_epoch=c10dg.data_size // batch_size + 1,
        epochs=epochs,
        validation_data=test_triplet_generator,
        validation_steps=c10dg.test_size // test_bs + 1,
        callbacks=[lrschedule, tboard, checkpoint]
    )

# History
with open(os.path.join(root_folder, "history.pkl"),"wb") as f:
    pickle.dump(history.history, f)
print("stored history to disk at ", os.path.join(root_folder, "history.pkl"))
