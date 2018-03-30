from keras import applications
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential, Model
from keras.layers import Dropout, Flatten, Dense, GlobalAveragePooling2D
from keras import backend as k
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard, EarlyStopping


#file_dir         = dirname(__file__)

img_width, img_height = 50, 75
train_data_dir = "data/train"
validation_data_dir = "data/val"
nb_train_samples = 4125
nb_validation_samples = 466
batch_size = 16
epochs = 50
nb_classes=16

#Import VGG19 model for transfer learning without output layers
vgg_model = applications.VGG19(weights = "imagenet", include_top=False, input_shape = (img_width, img_height, 3))

# Freeze the layers except the last 4
for layer in vgg_model.layers[:-4]:
    layer.trainable = False

# Check the layers
for layer in vgg_model.layers:
    print(layer, layer.trainable)

# Create the model
model = Sequential()

# Add the vgg model
model.add(vgg_model)

#Add output Layers
model.add(Flatten())
model.add(Dense(1024, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(1024, activation="relu"))
model.add(Dense(nb_classes, activation="softmax"))

model.summary()

from keras.utils import plot_model
#plot_model(model, to_file='model.png')