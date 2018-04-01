from keras import applications
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Model
from keras.layers import Dropout, Flatten, Dense, ZeroPadding2D, Input, Lambda
from keras.layers import Conv2D,Activation
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import MaxPooling2D
from keras import backend as K
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard, EarlyStopping

import tensorflow as tf


#file_dir         = dirname(__file__)

img_width, img_height = 50, 75
train_data_dir = "data/train"
validation_data_dir = "data/val"
nb_train_samples = 4125
nb_validation_samples = 466
batch_size = 16
epochs = 50
nb_classes=16
input_shape=(img_width, img_height, 3)

def hieroRecoModel_offline(input_shape,nb_classes):
    """
    Implementation of the Inception model used for FaceNet

    Arguments:
    input_shape -- shape of the images of the dataset

    Returns:
    model -- a Model() instance in Keras
    """
    # Define the input as a tensor with shape input_shape
    X_input = Input(input_shape)

    # Zero-Padding
    X = ZeroPadding2D((3, 3))(X_input)

    # First Block
    X = Conv2D(64, (7, 7), strides=(2, 2), name='conv1')(X)
    X = BatchNormalization(axis=1, name='bn1')(X)
    X = Activation('relu')(X)

    X = MaxPooling2D((3, 3), strides=2)(X)

    X = Flatten()(X)
    X = Dense(64, name='dense_layer')(X)

    # L2 normalization
    X = Lambda(lambda x: K.l2_normalize(x, axis=1))(X)


    # Create model instance
    model = Model(inputs=X_input, outputs=X, name='HieroRecoModel_off')

    return model


def hieroRecoModel(input_shape,nb_classes):
    """
    Implementation of the Inception model used for FaceNet

    Arguments:
    input_shape -- shape of the images of the dataset

    Returns:
    model -- a Model() instance in Keras
    """

    #Import VGG19 model for transfer learning without output layers
    vgg_model = applications.VGG19(weights = "imagenet", include_top=False, input_shape = input_shape)

    # Freeze the layers except the last 4
    for layer in vgg_model.layers[:-4]:
        layer.trainable = False

    # Check the layers
    for layer in vgg_model.layers:
        print(layer, layer.trainable)

    X_input = model.output

    # Adding custom Layers

    X = Flatten()(X_input)
    X = Dense(512, activation="relu")(X)
    X = Dropout(0.5)(X)
    X = Dense(128, activation="relu")(X)

    # L2 normalization
    X = Lambda(lambda x: K.l2_normalize(x, axis=1))(X)


    # Create model instance
    model = Model(inputs=vgg_model.input, outputs=X, name='HieroRecoModel')

    return model


def triplet_loss(anchor, positive, negative, alpha=0.2):
    """
    Implementation of the triplet loss as defined by formula (3)

    Arguments:
    y_true -- true labels, required when you define a loss in Keras, you don't need it in this function.
    y_pred -- python list containing three objects:
            anchor -- the encodings for the anchor images, of shape (None, 128)
            positive -- the encodings for the positive images, of shape (None, 128)
            negative -- the encodings for the negative images, of shape (None, 128)

    Returns:
    loss -- real number, value of the loss
    """

    #anchor, positive, negative = y_pred[0], y_pred[1], y_pred[2]

    ### START CODE HERE ### (≈ 4 lines)
    # Step 1: Compute the (encoding) distance between the anchor and the positive, you will need to sum over axis=-1
    pos_dist = tf.reduce_sum(tf.square(anchor - positive), axis=-1)
    # Step 2: Compute the (encoding) distance between the anchor and the negative, you will need to sum over axis=-1
    neg_dist = tf.reduce_sum(tf.square(anchor - negative), axis=-1)
    # Step 3: subtract the two previous distances and add alpha.
    # basic_loss = tf.add(tf.subtract(pos_dist,neg_dist),alpha)
    basic_loss = tf.add(tf.subtract(pos_dist, neg_dist), alpha)
    # Step 4: Take the maximum of basic_loss and 0.0. Sum over the training examples.
    loss = tf.reduce_sum(tf.maximum(basic_loss, 0))
    ### END CODE HERE ###

    return loss


def triplet_loss_2(anchor, positive, negative, alpha=0.4):
    """
    Implementation of the triplet loss function
    Arguments:
    y_true -- true labels, required when you define a loss in Keras, you don't need it in this function.
    y_pred -- python list containing three objects:
            anchor -- the encodings for the anchor data
            positive -- the encodings for the positive data (similar to anchor)
            negative -- the encodings for the negative data (different from anchor)
    Returns:
    loss -- real number, value of the loss
    """

    #anchor = y_pred[:, 0:3]
    #positive = y_pred[:, 3:6]
    #negative = y_pred[:, 6:9]

    # distance between the anchor and the positive
    pos_dist = K.sum(K.square(anchor - positive), axis=1)

    # distance between the anchor and the negative
    neg_dist = K.sum(K.square(anchor - negative), axis=1)

    # compute loss
    basic_loss = pos_dist - neg_dist + alpha
    loss = K.maximum(basic_loss, 0.0)

    return loss


FRmodel = hieroRecoModel_offline(input_shape=input_shape,nb_classes=nb_classes)
FRmodel.summary()

FRmodel.compile(optimizer = 'adam', loss = triplet_loss, metrics = ['accuracy'])