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
    if len(sys.argv) == 1:
        print("WARNING: No checkpoint specified!, using random weights")
    else: 
        print("Loading weights from location: ", sys.argv[1])
        wts_location = sys.argv[1] 
        model.load_weights(wts_location, by_name=True)  #Load weights 

    
    # import ipdb; ipdb.set_trace()
    # Dataset 
    (trainX, trainY), (testX, testY) = cifar10.load_data()
    trainY = to_categorical(trainY)
    testY = to_categorical(testY)
    # Test dataset 
    test_dgen = ImageDataGenerator(
                    featurewise_center=True, 
                    featurewise_std_normalization=True
                )
    test_dgen.fit(trainX) # IMP! mean,std calculated on training data

    # FGSM parameters 
    eta = 0.007
    if len(sys.argv) == 1:
        print("Setting default value of eta...")
    else:
        eta = float(sys.argv[2])

    # Performance before attack 
    preds_pre_attack = model.predict_generator(
                            generator=test_dgen.flow(testX, testY, batch_size=50),
                            steps=len(testX)/50,
                            verbose=1
                        )
    print("Performance before attack is: ", 
            np.count_nonzero(np.argmax(preds_pre_attack,axis=1) == np.argmax(testY,axis=1)),
            " correct out of 10000 test samples"
        )

    ### Perform attack 
    # Calculate grad w.r.t input for all test images 
    print("Calculating gradient w.r.t input..")
    calc_grads = create_gradient_function(model, 0, 0)
    grads_X_test = [] 
    for batch_idx in tqdm(range(0, len(testX), 50)):
        x_batch = testX[batch_idx:batch_idx+50]
        y_batch = testY[batch_idx:batch_idx+50] 
        _, grads_batch = calc_grads([x_batch, y_batch]) 
        grads_X_test.append(grads_batch)
    grads_X_test = np.concatenate(grads_X_test, axis=0) 

    # attacked = orig + eta*sign(grad)
    assert grads_X_test.shape == testX.shape 
    attacked_testX = testX + eta*np.sign(grads_X_test)

    # Performance after attack 
    preds_post_attack = model.predict_generator(
                            generator=test_dgen.flow(attacked_testX, testY, batch_size=50),
                            steps=len(attacked_testX)/50,
                            verbose=1
                        )
    print("Performance after attack is: ", 
            np.count_nonzero(np.argmax(preds_post_attack,axis=1) == np.argmax(testY,axis=1)),
            " correct out of 10000 test samples"
        )


    # x_val = np.random.randn(1,32,32,3)
    # y_val = np.zeros((1,10))

    # loss_val, grad_val = get_gradients_function(inputs=(x_val, y_val))
    # # Now, we have the grads w.r.t input. 

    # import ipdb; ipdb.set_trace()


