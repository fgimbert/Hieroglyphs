import os
import pandas
import numpy as np
import random
#from PIL import Image

import matplotlib.pyplot as plt

import numpy as np
from scipy.ndimage import imread

#from skimage import io

from keras import applications
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Model
from keras.layers import Dropout, Flatten, Dense, ZeroPadding2D, Input, Lambda, concatenate
from keras.layers import Conv2D,Activation
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import MaxPooling2D
from keras.optimizers import Adam, RMSprop, SGD

from keras import backend as K
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard, EarlyStopping

import tensorflow as tf


path="/Users/fgimbert/Documents/Dataset/Manual/Preprocessed/"


def loadData(folderPictures=path):

    folders=next(os.walk(folderPictures))[1]

    img_groups = {}
    img_list={}

    for folder in folders:
        for img_file in os.listdir(folderPictures+folder):
            name, label = img_file.strip('.png').split("_")

            # One image per class

            #if label not in img_groups.keys():
            #    img_groups[label] = [folder + "_" + name]


            # Multiple images per class

            if label in img_groups.keys():
                img_groups[label].append(folder+"_"+name)
            else:
                img_groups[label] = [folder+"_"+name]

            img_list[folder+"_"+name]=[label]


    # Remove class with only one hieroglyph


    for k,v in list(img_groups.items()):
        if len(v)==1: del img_groups[k]

    # Extract only N hieroglyph classes randomly

    nclass = 50

    list_of_class = random.sample(list(img_groups.keys()), nclass)
    #print(list_of_class)

    short_dico = {x: img_groups[x] for x in list_of_class if x in img_groups}





    dataHiero=pandas.DataFrame.from_dict(img_list,orient='index')
    dataHiero.columns = ["label"]
    dataHiero = dataHiero[dataHiero.label != 'UNKNOWN']

    dataHiero = dataHiero.loc[dataHiero['label'].isin(short_dico)]


    dataHiero.reset_index(level=0, inplace=True)

    return dataHiero,img_groups

def loadTriplets(dataset,labels):

    N_hieros=len(dataset)

    tripletHiero=[]

    for i in range(N_hieros):
        label=dataHiero['label'][i]
        hiero=dataHiero['index'][i]

        pos_hiero=labels.setdefault(label)
        positive=hiero

        while positive==hiero:
            positive=random.choice(pos_hiero)

        if positive==hiero: print('Positive Choice Error ! ')


        neg_label=label
        neg_labels=list(labels.keys())

        while neg_label==label or neg_label=='UNKNOWN':
            neg_label = random.choice(neg_labels)

        negative=random.choice(labels[neg_label])

        #if negative == hiero : print('Negative Choice Error ! ')

        tripletHiero.append([hiero,positive,negative,label,neg_label])
        dataTriplet =pandas.DataFrame(tripletHiero,columns=['anchor','positive','negative','label','neg_label'])



    return dataTriplet


def loadPictures(data):

    N_hieros = len(data)
    repertory, file = data['anchor'][0].split("_")
    label=str(data['label'][0])

    picture="/Users/fgimbert/Documents/Dataset/Manual/Preprocessed/"+str(repertory)+"/"+str(file)+"_"+label+".png"

    #im = Image.open(picture)


    #img_x=im.size[0]
    #img_y=im.size[1]

    img_x=50
    img_y=75

    anchor, positive, negative = np.zeros((N_hieros,img_x*img_y)),np.zeros((N_hieros,img_x*img_y)),np.zeros((N_hieros,img_x*img_y))
    labels_true = []
    labels_wrong= []


    for index, row in data.iterrows():

        repertory, file = row['anchor'].split("_")
        label = row['label']
        picture = "/Users/fgimbert/Documents/Dataset/Manual/Preprocessed/" + str(repertory) + "/" + str(
            file) + "_" + str(label) + ".png"
        labels_true.append(label)
        anchor[index]=imread(picture, flatten=True).reshape(1,img_x*img_y)

        repertory, file = row['positive'].split("_")
        picture = "/Users/fgimbert/Documents/Dataset/Manual/Preprocessed/" + str(repertory) + "/" + str(
            file) + "_" + str(label) + ".png"
        positive[index] = imread(picture, flatten=True).reshape(1, img_x * img_y)

        repertory, file = row['negative'].split("_")
        label = row['neg_label']
        picture = "/Users/fgimbert/Documents/Dataset/Manual/Preprocessed/" + str(repertory) + "/" + str(
            file) + "_" + str(label) + ".png"
        labels_wrong.append(label)
        negative[index] = imread(picture, flatten=True).reshape(1, img_x * img_y)

    return [anchor,positive,negative],labels_true,labels_wrong

def triplet_loss(y_pred, alpha=0.5):

    """
    Implementation of the triplet loss as defined by formula (3)

    y_pred -- python list containing three objects:
            anchor -- the encodings for the anchor images, of shape (None, 128)
            positive -- the encodings for the positive images, of shape (None, 128)
            negative -- the encodings for the negative images, of shape (None, 128)

    Returns:
    loss -- real number, value of the loss
    """

    anchor, positive, negative = y_pred[0], y_pred[1], y_pred[2]
    pos_dist = tf.reduce_sum(tf.square(anchor - positive), axis=-1)
    neg_dist = tf.reduce_sum(tf.square(anchor - negative), axis=-1)
    basic_loss = tf.add(tf.subtract(pos_dist, neg_dist), alpha)
    loss = tf.reduce_sum(tf.maximum(basic_loss, 0))

    return loss



def hieroRecoModel_offline(input_shape):
    """
    Arguments:
    input_shape -- shape of the images of the dataset

    Returns:
    model -- a Model() instance in Keras
    """
    X_input = Input(input_shape)

    # Zero-Padding
    X = ZeroPadding2D((3, 3))(X_input)

    # First Block
    X = Conv2D(64, (3, 3), strides=(2, 2), name='conv1')(X)
    X = BatchNormalization(axis=1, name='bn1')(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((3, 3), strides=2)(X)

    X = Conv2D(64, (3, 3))(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((3, 3), strides=2)(X)

    X = Flatten()(X)
    X = Dense(128, name='dense_layer')(X)

    # L2 normalization
    X = Lambda(lambda x: K.l2_normalize(x, axis=1))(X)

    features = Model(X_input, X, name="features")

    # Inputs of the siamese network

    anchor = Input(shape=input_shape)
    positive = Input(shape=input_shape)
    negative = Input(shape=input_shape)

    # Embedding Features of input

    anchor_features = features(anchor)
    pos_features = features(positive)
    neg_features = features(negative)

    input_triplet = [anchor, positive, negative]
    output_features = [anchor_features, pos_features, neg_features]

    # Define the trainable model
    loss_model = Model(inputs=input_triplet, outputs=output_features,name='loss')
    loss_model.add_loss(K.mean(triplet_loss(output_features)))
    loss_model.compile(loss=None,optimizer='adam')


    # Create model instance
    #model = Model(inputs=X_input, outputs=X, name='HieroRecoModel_off')

    return features, loss_model


def hieroRecoModel_online(input_shape):
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
    #model = Model(inputs=vgg_model.input, outputs=X, name='HieroRecoModel')
    features = Model(vgg_model.input, X, name="features")

    # Inputs of the siamese network

    anchor = Input(shape=input_shape)
    positive = Input(shape=input_shape)
    negative = Input(shape=input_shape)

    # Embedding Features of input

    anchor_features = features(anchor)
    pos_features = features(positive)
    neg_features = features(negative)

    input_triplet = [anchor, positive, negative]
    output_features = [anchor_features, pos_features, neg_features]

    # Define the trainable model
    loss_model = Model(inputs=input_triplet, outputs=output_features, name='loss')
    loss_model.add_loss(K.mean(triplet_loss(output_features)))
    loss_model.compile(loss=None, optimizer='adam')

    # Create model instance
    # model = Model(inputs=X_input, outputs=X, name='HieroRecoModel_off')

    return features, loss_model



img_width, img_height = 50, 75
input_shape=(img_width, img_height, 1)

dataHiero,dictLabels=loadData(path)

print(dataHiero.head())
print(len(dataHiero)," hieroglyphs !")
print(len(dictLabels.keys())," different hieroglyphs  !")

tripletData=loadTriplets(dataHiero,dictLabels)
dataset,labels_true,labels_wrong =loadPictures(tripletData)

if dataset[0].shape[0]>1500:
    ntrain=1500
else:
    ntrain=1000


train_data=[dataset[0][:ntrain].reshape((-1,img_width, img_height, 1)),dataset[1][:ntrain].reshape((-1,img_width, img_height, 1)),dataset[2][:ntrain].reshape((-1,img_width, img_height, 1))]
test_data=dataset[0][ntrain:].reshape((-1,img_width, img_height, 1))

print(train_data[0].shape)



features, loss_model = hieroRecoModel_offline(input_shape)
loss_model.summary()
history = loss_model.fit(train_data,batch_size=16,epochs=15,verbose=2)


#plt.plot(history.history['loss'],label='loss')
#plt.legend()
#plt.savefig('loss.png')


def encoding_database(data,model):

    database=np.zeros((data.shape[0],128))
    shapedata=data.shape
    print(shapedata)

    for index,row in enumerate(data):
        row=row.reshape(1,shapedata[1],shapedata[2],shapedata[3])
        #print(row.shape)
        X_features=model.predict(row)
        #print(X_features.shape)
        database[index]=X_features

    return database

train_hiero=encoding_database(train_data[0],features)
test_hiero=encoding_database(test_data,features)

dico_hiero={}

for index in range(ntrain):
    if labels_true[index] not in dico_hiero.keys():

        dico_hiero[labels_true[index]] = train_hiero[index]

print(dico_hiero.keys())



def which_hiero(image,dico_hiero):

    min_dist=100

    for (name,db_enc) in dico_hiero.items():
        dist=np.linalg.norm(image-db_enc)

        if dist<min_dist:
            min_dist = dist
            identity=name

    #if min_dist>0.5:
        #identity='UNKNOWN'

    return min_dist, identity

#####TEST RECOGNITION#####

for i in range(10):
    dist, hieroglyph = which_hiero(test_hiero[i], dico_hiero)
    print("True Hieroglyph : " ,labels_true[ntrain+i],"// Predicted : " ,hieroglyph, "dist : ", dist)

