from keras.callbacks import TensorBoard, ReduceLROnPlateau, Callback
from keras.layers import Input, Conv2D, MaxPooling2D, Dropout, Flatten, Lambda, Dense 
from keras.models import Sequential 
import keras.backend as K 
from keras import optimizers 
from keras.datasets import mnist 
from loss_layers import triplet_loss, triplet_loss_batched_wrapper

import numpy as np 
import time 
import matplotlib.pyplot as plt 
from sklearn.utils import shuffle
from sklearn.manifold import TSNE 

def nw_arch_mnist():
    """
    LeNet architecture 
    """
    EMBEDDING_UNITS = 128 
    input_shape = (28,28,1)
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),
                     activation='relu',
                     input_shape=input_shape))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(EMBEDDING_UNITS))
    model.add(Lambda(lambda x: K.l2_normalize(x, axis=-1))) # normalize each row individually. 
    return model 

"""
PREP DATA
"""
EMBEDDING_UNITS = 128
def triplet_generator():
    (x_train, y_train), (x_test, y_test) = mnist.load_data() 
    x_train, y_train = shuffle(x_train, y_train)

    while True: 

        anc_class = np.random.randint(0, 10) 
        neg_class = None 
        while neg_class in [anc_class, None]:
            neg_class = np.random.randint(0, 10) 
        # checks 
        assert neg_class != anc_class 

        anc = x_train[y_train==anc_class]
        anc,pos = anc[np.random.choice(anc.shape[0], size=2, replace=False)]
        neg = x_train[y_train==neg_class] 
        neg = neg[np.random.choice(neg.shape[0], size=1)][0]

        yield anc, pos, neg

def batched_triplet_generator(batch_size):
    tgen = triplet_generator() 
    while True:
        L_anc, L_pos, L_neg = [], [], [] 
        for _ in range(batch_size):
            anc, pos, neg = next(tgen) 
            L_anc.append(anc); L_pos.append(pos); L_neg.append(neg)

        batch = np.vstack([np.array(L_anc), np.array(L_pos), np.array(L_neg)]) 
        batch = np.expand_dims(batch, axis=3)
        yield  batch, np.zeros((batch_size*3,EMBEDDING_UNITS))

def _test_generator():

    dgen = batched_triplet_generator(batch_size=32)
    data, _ = next(dgen) 

    for idx in range(8):
        anc_idx = np.random.randint(0, 32) 
        catted  = np.hstack( [data[anc_idx,:,:,0],data[anc_idx+32,:,:,0], data[anc_idx+32+32,:,:,0]] ) 
        plt.subplot(8,1,idx+1) 
        plt.imshow(catted) 
    plt.show()

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
custom_loss = triplet_loss_batched_wrapper(num_triplets=16) 
model.compile(optimizer="sgd", loss=custom_loss)

valcb = valid_callback(top_k=100, num_samples=400)
lrreduce = ReduceLROnPlateau(monitor="loss", factor=0.1, patience=4, verbose=1, min_lr=1e-08)

dgen = batched_triplet_generator(batch_size=16)

history = model.fit_generator(
        dgen, 
        steps_per_epoch=500,
        epochs=50, 
        callbacks=[valcb, lrreduce]
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





