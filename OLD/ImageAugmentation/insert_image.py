import os
import pandas
import numpy as np
import random
#from PIL import Image
from skimage.color import gray2rgb


import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import numpy as np


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
import pylab as plt
import numpy as np
from PIL import Image

#path="/Users/fgimbert/Documents/Dataset/Manual/Preprocessed/"
path="../../HieroDataset/Manual/Preprocessed/"

i=0
folders=next(os.walk(path))[1]
print(folders)

for folder in folders[:-2]:
    print('folder', folder)
    for img_file in os.listdir(path + folder):
        #print(img_file)
        name, label = img_file.strip('.png').split("_")
        Z = np.random.rand(500, 500, 3) * 255  # Test data
        background = Image.fromarray(Z.astype('uint8')).convert('L')

        #print(background.size)
        # image = mpimg.imread(path+fold +'/' +img)

        image = Image.open(path + folder +'/' + img_file)
        #print(folder + img_file)
        #print(image.size)
        img_w, img_h = image.size

        bg_w, bg_h = background.size
        offset = ((bg_w - img_w) // 2, (bg_h - img_h) // 2)

        background.paste(image, (np.random.randint(0, (bg_w - img_w) - 1), np.random.randint(0, (bg_h - img_h) - 1)))
        background.save('./augmented_train/img_{0}.png'.format(i))
        i += 1

i = 0
for folder in folders[-1:]:
    print('folder', folder)
    for img_file in os.listdir(path + folder):
        #print(img_file)
        name, label = img_file.strip('.png').split("_")
        Z = np.random.rand(500, 500, 3) * 255  # Test data
        background = Image.fromarray(Z.astype('uint8')).convert('L')

        #print(background.size)
        # image = mpimg.imread(path+fold +'/' +img)

        image = Image.open(path + folder +'/' + img_file)
        #print(folder + img_file)
        #print(image.size)
        img_w, img_h = image.size

        bg_w, bg_h = background.size
        offset = ((bg_w - img_w) // 2, (bg_h - img_h) // 2)

        background.paste(image, (np.random.randint(0, (bg_w - img_w) - 1), np.random.randint(0, (bg_h - img_h) - 1)))
        background.save('./augmented_test/img_{0}.png'.format(i))
        i += 1

