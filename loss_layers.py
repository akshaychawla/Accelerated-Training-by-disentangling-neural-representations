from __future__ import print_function
import keras.backend as K
from keras.layers import Lambda, concatenate

def triplet_loss(y_true, y_pred):
    """
    y_true : FAKE
    y_pred : (3,embedding_units) vector
    """
    print("\n\n\t\tYOU SHOULD NOT BE USING def:triplet_loss!!!!\n\n")
    alpha = 0.1
    anchor = y_pred[0, :]
    positive = y_pred[1,:]
    negative = y_pred[2,:]
    loss = K.sqrt(K.sum(K.square(anchor-positive))) - K.sqrt(K.sum(K.square(anchor-negative))) + alpha
    loss = K.maximum(0.0, loss)
    return loss


def triplet_loss_batched_wrapper(num_triplets):
    """
    num_triplets is the number of triplets that will be given by the
    network output.
    i.e the network output y_pred will be of shape (num_triplets*3, embedding_units)
    Using a Closure type approach
    """
    def triplet_loss_batched(y_true, y_pred):
        """
        y_true : FAKE
        y_pred : (num_triplets*3, embedding_units) matrix
        """
        alpha = 0.1
        anchors = y_pred[0: num_triplets, :]
        positives = y_pred[num_triplets: num_triplets*2, :]
        negatives = y_pred[num_triplets*2: num_triplets*3, :]
        loss_per_sample = K.sqrt(K.sum(K.square(anchors-positives), axis=-1)) - K.sqrt(K.sum(K.square(anchors-negatives), axis=-1)) + alpha
        loss_per_sample = K.maximum(0.0, loss_per_sample)
        loss_batch = K.mean(loss_per_sample, axis=0)
        return loss_batch

    return triplet_loss_batched


def wrapper_categorical_crossentropy(num_triplets):
    """
    y_true : one-hot (anc*n, pos*n, neg*n) X 10
    y_pred : softmax (anc*n, pos*n, neg*n) X 10

    Since the anchor and positive vectors are of the same class,
    this function isolates the anchor and negative instances and
    uses them for training.
    """
    select_anchors = lambda x: x[:num_triplets, :]
    select_negatives = lambda x: x[num_triplets*2 : num_triplets*3, :]
    select_positives = lambda x: x[num_triplets : num_triplets*2, :] 

    def custom_cat_ce(y_true, y_pred):
        true_ancs = Lambda(select_anchors)(y_true)
        true_negs = Lambda(select_negatives)(y_true)
        true_poss = Lambda(select_positives)(y_true)

        pred_ancs = Lambda(select_anchors)(y_pred)
        pred_negs = Lambda(select_negatives)(y_pred)
        pred_poss = Lambda(select_positives)(y_pred) 

        return K.categorical_crossentropy(
            concatenate([true_ancs, true_negs, true_poss]),
            concatenate([pred_ancs, pred_negs, pred_poss])
        )

    return custom_cat_ce
