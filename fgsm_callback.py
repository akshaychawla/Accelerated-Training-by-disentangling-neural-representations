"""
Callback for fgsm. 
This will attack the model after each epoch
"""
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
    assert "softmax" in output_tensor.name.lower(), "[ERROR] output tensor name is ",output_tensor.name()
    loss_tensor   = losses.categorical_crossentropy(gt_tensor, output_tensor)
    grads_tensor  = K.gradients(loss_tensor, input_tensor)[0]

    get_gradients_function = K.function(
                                inputs=[input_tensor, gt_tensor],
                                outputs=[loss_tensor, grads_tensor])
    return get_gradients_function

class fgsm_callback(Callback):

    def __init__(self, eta=0.05):
        super(fgsm_callback, self).__init__() 
        self.eta = eta

        # data
        (self.trainX, self.trainY), (self.testX, self.testY) = cifar10.load_data()
        self.trainY = to_categorical(self.trainY)
        self.testY = to_categorical(self.testY)

        # generators
        test_dgen = ImageDataGenerator(
                        featurewise_center=True,
                        featurewise_std_normalization=True
                    )
        test_dgen.fit(self.trainX) # IMP! mean,std calculated on training data

        # normalize test set
        print("Generating normed version of test set..")
        testX_normed = [] 
        temp_testdgen = test_dgen.flow(self.testX, self.testY, batch_size=50, shuffle=False)
        for _ in range(len(self.testX)//50):
            x_batch, y_batch = next(temp_testdgen)
            testX_normed.append(x_batch)
        testX_normed = np.concatenate(testX_normed, axis=0)
        del temp_testdgen, x_batch, y_batch
        print("Done")
        self.testX_normed = testX_normed

    def on_epoch_end(self, epoch, logs = {}):
        """
        Attack!
        """

        # Performance before attack
        performance_pre_attack = self.model.evaluate(
                                x = self.testX_normed, 
                                y = self.testY, 
                                batch_size=50,
                                verbose=1
                            )
        print("Accuracy before attack is: ", performance_pre_attack[1])

        ### Perform attack
        # Calculate grad w.r.t input for all test images
        print("Calculating gradient w.r.t input..")
        calc_grads = create_gradient_function(self.model, 0, 0)
        grads_X_test = []
        for batch_idx in tqdm(range(0, len(self.testX_normed), 50)):
            x_batch = self.testX_normed[batch_idx: batch_idx+50]
            y_batch = self.testY[batch_idx: batch_idx+50]
            _, grads_batch = calc_grads([x_batch, y_batch])
            grads_X_test.append(grads_batch)
        grads_X_test = np.concatenate(grads_X_test, axis=0)

        # attacked = orig + eta*sign(grad)
        assert grads_X_test.shape == self.testX_normed.shape
        attacked_testX = self.testX_normed + eta*np.sign(grads_X_test)

        # Performance after attack
        performance_post_attack = self.model.evaluate(
                                x = attacked_testX, 
                                y = self.testY, 
                                batch_size = 50,
                                verbose = 1
                            )
        print("Accuracy after attack is: ", performance_post_attack[1])



