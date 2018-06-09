import numpy as np
import matplotlib.pyplot as plt
import os, sys, time, pickle
import wide_residual_network as wrn
from keras.datasets import cifar10
from keras.callbacks import Callback, ModelCheckpoint, LearningRateScheduler, TensorBoard
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD

# mode
mode = sys.argv[1].lower()
print("\n\n\t\tTRAINING MODE: %s\n\n"%mode)
if mode == "triplet":
    root_folder = "./TRIP_{}/".format(time.time())
elif mode == "normal":
    root_folder = "./RUN_{}/".format(time.time())
else:
    print("\n\n\t\tINCORRECT MODE ARG RECEIVED. EXITING.\n\n")
    sys.exit()

# hyperparameters
batch_size = 128
epochs   = 200
img_rows, img_cols = 32, 32
weight_decay = 0.0005

# Logs + checkpoints directory
os.makedirs(root_folder)
os.makedirs(root_folder+"logs")
os.makedirs(root_folder+"checkpoints")
print("Created folders in .. ",root_folder)

# Dataset
(trainX, trainY), (testX, testY) = cifar10.load_data()
trainY = to_categorical(trainY)
testY = to_categorical(testY)

# Data generators
train_dgen = ImageDataGenerator(
                featurewise_center=True,
                featurewise_std_normalization=True,
                horizontal_flip=True,
                width_shift_range=4, # mimick padding=4 + randomcrop
                height_shift_range=4,
                fill_mode="nearest"
            )
train_dgen.fit(trainX)
test_dgen = ImageDataGenerator(
                featurewise_center=True,
                featurewise_std_normalization=True
            )
test_dgen.fit(trainX) # IMP! mean,std calculated on training data

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
model = wrn.create_wide_residual_network(
            (32,32,3),
            nb_classes=10,
            N=4, k=10,
            dropout=0.0,
            mode=mode
        )
print(model.summary())
sgd = SGD(lr=0.1, momentum=0.9, nesterov=True)
model.compile(loss="categorical_crossentropy", optimizer=sgd, metrics=["accuracy"])

# Train
history = model.fit_generator(
        train_dgen.flow(trainX, trainY, batch_size=batch_size),
        steps_per_epoch=len(trainX) // batch_size + 1,
        validation_data=test_dgen.flow(testX, testY, batch_size=batch_size),
        validation_steps=len(testX) // batch_size + 1,
        epochs=epochs,
        callbacks=[lrschedule, tboard, checkpoint]
    )
with open(os.path.join(root_folder, "history.pkl"),"wb") as f:
    pickle.dump(history.history, f)
print("stored history to disk at ", os.path.join(root_folder, "history.pkl"))
