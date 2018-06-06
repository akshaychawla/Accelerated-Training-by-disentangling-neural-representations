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

# Network 
model = wrn.create_wide_residual_network(
            (32,32,3),
            nb_classes=10, 
            N=4, k=10, 
            dropout=0.0
        )
# check_location = sys.argv[1] 
# model.load_weights(check_location, by_name=True)  #Load weights 

gt_tensor     = K.placeholder(shape=(None, 10))
input_tensor  = model.input 
output_tensor = model.output 
loss_tensor   = losses.categorical_crossentropy(gt_tensor, output_tensor) 
grads_tensor  = K.gradients(loss_tensor, input_tensor)[0]

get_gradients_function = K.function(
                            inputs=[input_tensor, gt_tensor], 
                            outputs=[loss_tensor, grads_tensor]
                        )

x_val = np.random.randn(1,32,32,3)
y_val = np.zeros((1,10))

loss_val, grad_val = get_gradients_function(inputs=(x_val, y_val))
# Now, we have the grads w.r.t input. 

import ipdb; ipdb.set_trace()


