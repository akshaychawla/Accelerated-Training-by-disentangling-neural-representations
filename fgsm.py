import numpy as np
import matplotlib.pyplot as plt
import os, sys, time, pickle
import wide_residual_network as wrn
from keras.datasets import cifar10
from keras.callbacks import Callback, ModelCheckpoint, LearningRateScheduler, TensorBoard
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD
import keras.backend as K
from keras import losses
from tqdm import *

def create_gradient_function(model, input_idx, output_idx):
    gt_tensor     = K.placeholder(shape=(None, 10))
    input_tensor  = model.inputs[input_idx]
    output_tensor = model.outputs[output_idx]
    loss_tensor   = losses.categorical_crossentropy(gt_tensor, output_tensor)
    grads_tensor  = K.gradients(loss_tensor, input_tensor)[0]

    get_gradients_function = K.function(
                                inputs=[input_tensor, gt_tensor],
                                outputs=[loss_tensor, grads_tensor])
    return get_gradients_function


if __name__ == "__main__":

    # Network
    model = wrn.create_wide_residual_network(
                (32,32,3),
                nb_classes=10,
                N=4, k=10,
                dropout=0.0
            )

    # load wts
    model.compile(loss="categorical_crossentropy", optimizer="sgd", metrics=["accuracy"])
    if len(sys.argv) == 1:
        print("WARNING: No checkpoint specified!, using random weights")
    else:
        print("Loading weights from location: ", sys.argv[1])
        wts_location = sys.argv[1]
        model.load_weights(wts_location, by_name=True)  #Load weights

        # Load the final dense layer (preds) weights correctly 
        import h5py 
        f = h5py.File(wts_location,"r") 
        all_layers = list(f.keys())
        if "preds" in all_layers:
            print("Found name preds in ", wts_location)
            preds_wts = [ 
                    f["preds"]["preds"]["kernel:0"][:,:], 
                    f["preds"]["preds"]["bias:0"][:]
                ]
            model.layers[-1].set_weights(preds_wts)
        else:
            print("..Could not find layer by name of preds in", wts_location)
        f.close()


    # import ipdb; ipdb.set_trace()
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

    # Generate normed version of dataset 
    print("Generating normed version of test set..")
    testX_normed = [] 
    temp_testdgen = test_dgen.flow(testX, testY, batch_size=50, shuffle=False)
    for _ in range(len(testX)//50):
        x_batch, y_batch = next(temp_testdgen)
        testX_normed.append(x_batch)
    testX_normed = np.concatenate(testX_normed, axis=0)
    del temp_testdgen, x_batch, y_batch
    print("Done")

    # FGSM parameters
    # import ipdb; ipdb.set_trace()
    eta = 0.007
    if len(sys.argv) == 1:
        print("Setting default value of eta...")
    else:
        eta = float(sys.argv[2])
        print("Using eta = ", eta)

    # Performance before attack
    performance_pre_attack = model.evaluate(
                            x = testX_normed, 
                            y = testY, 
                            batch_size=50,
                            verbose=1
                        )
    print("Accuracy before attack is: ", performance_pre_attack[1])

    ### Perform attack
    # Calculate grad w.r.t input for all test images
    print("Calculating gradient w.r.t input..")
    calc_grads = create_gradient_function(model, 0, 0)
    grads_X_test = []
    for batch_idx in tqdm(range(0, len(testX_normed), 50)):
        x_batch = testX_normed[batch_idx: batch_idx+50]
        y_batch = testY[batch_idx: batch_idx+50]
        _, grads_batch = calc_grads([x_batch, y_batch])
        grads_X_test.append(grads_batch)
    grads_X_test = np.concatenate(grads_X_test, axis=0)

    # attacked = orig + eta*sign(grad)
    assert grads_X_test.shape == testX_normed.shape
    attacked_testX = testX_normed + eta*np.sign(grads_X_test)

    # Performance after attack
    performance_post_attack = model.evaluate(
                            x = attacked_testX, 
                            y = testY, 
                            batch_size = 50,
                            verbose = 1
                        )
    print("Accuracy after attack is: ", performance_post_attack[1])
