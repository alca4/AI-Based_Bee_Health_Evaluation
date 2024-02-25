import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import imageio
import pickle

from skimage.transform import rescale, resize, rotate
from skimage.color import rgb2gray
from sklearn.metrics import confusion_matrix, auc, accuracy_score
from sklearn.model_selection import train_test_split

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam, RMSprop, SGD
from keras import callbacks
from keras.models import load_model
from keras import models

import warnings
warnings.filterwarnings("ignore")


random_state = 42

np.random.seed(random_state)

raw_data  = pd.read_csv('./Downloads/Bee_Images_Health_Merge_Class4_Out.csv')
raw_data.head()

raw_data.shape

health_counts = raw_data["health"].value_counts()

plt.title("Counts of bee $health$ categories")
g = sns.barplot(x = health_counts, y = health_counts.index);
g.set_xlabel("Frequency");

# Clear unnecessary columns
data = raw_data[["file","health"]]
data.head(2)

def get_image_data(files):
    '''Returns np.ndarray of images read from the image data directory'''
    IMAGE_FILE_ROOT = './Downloads/Bee_Images_Health_Merge_Class4_Out3/'
    return np.asanyarray([imageio.imread("{}{}".format(IMAGE_FILE_ROOT, file)) for file in files])

def show_image(image, ax = plt, title = None, show_size = False):
    '''Plots a given np.array image'''
    ax.imshow(image)
    if title:
        if ax == plt:
            plt.title(title)
        else:
            ax.set_title(title)
    if not show_size:
        ax.tick_params(bottom = False, left = False, labelbottom = False, labelleft = False)

raw_images = get_image_data(data["file"].values)
show_image(raw_images[100])

def show_images(images, titles = None, show_size = False):
    '''Plots many images from the given list of np.array images'''
    cols = 4
    f, ax = plt.subplots(nrows=int(np.ceil(len(images)/cols)),ncols=cols, figsize=(14,5))
    ax = ax.flatten()
    for i, image in enumerate(images):
        if titles:
            show_image(image, ax = ax[i], title = titles[i], show_size = show_size)
        else:
            show_image(image, ax = ax[i], title = None, show_size = show_size)
    plt.show()

def get_images_wh(images):
    '''Returns a tuple of lists, representing the widths and heights of the give images, respectively.'''
    widths = []
    heights = []
    for image in images:
        h, w, rbg = image.shape
        widths.append(w)
        heights.append(h)
    return (widths, heights)

def get_best_average(dist, cutoff = .5):
    '''Returns an integer of the average from the given distribution above the cutoff.'''
    # requires single peak normal-like distribution
    hist, bin_edges = np.histogram(dist, bins = 25);
    total_hist = sum(hist)

    # associating proportion of hist with bin_edges
    hist_edges = [(vals[0]/total_hist,vals[1]) for vals in zip(hist, bin_edges)]

    # sorting by proportions (assumes normal-like dist such that high freq. bins are close together)
    hist_edges.sort(key = lambda x: x[0])
    lefts = []

    # add highest freq. bins to list up to cutoff % of total
    while cutoff > 0:
        vals = hist_edges.pop()
        cutoff -= vals[0]
        lefts.append(vals[1])

    # determining leftmost and rightmost range, then returning average
    diff = np.abs(np.diff(lefts)[0]) # same diff b/c of bins
    leftmost = min(lefts)
    rightmost = max(lefts) + diff
    return int(np.round(np.mean([rightmost,leftmost])))

wh = get_images_wh(raw_images)

size = 18
plt.title("Widths of bee images", fontsize = size * 4/3, pad = size/2)
plt.ylabel("Frequency", size = size)
plt.xlabel("width (pixels)", size = size)
plt.hist(wh[0], bins = 25);

size = 18
plt.title("Heights of bee images", fontsize = size * 4/3, pad = size/2)
plt.ylabel("Frequency", size = size)
plt.xlabel("width (pixels)", size = size)
plt.hist(wh[1], bins = 25);

IDEAL_WIDTH, IDEAL_HEIGHT = 299, 299

data["health"].value_counts(normalize = True)

def normalize(image):
    return (image/255. - 0.5)

def create_datagens(
    data, datagen_params,
    target_shape, batch_size, x_col="file", y_col="health", IMAGE_FILE_ROOT = './Downloads/Bee_Images_Health_Merge_Class4_Out3/',
    random_state = 42, preprocessing_function = None):
        '''
        Appropriately creates and returns two ImageDataGenerator objects - one for training and one for testing.

        ImageDataGenerator objects are responsible for handling image data during model training, by pulling the data
        directly from the image data directory, resizing the image, and applying the appropriate transformations.

        The testing ImageDataGenerator object does not apply any transformations.
        '''
        data[y_col] = data[y_col].astype(str)
        # train/test split
        train, test = train_test_split(
            data,
            test_size = 0.2,
            stratify = data.iloc[:,-1], # last column is target variable
            random_state = random_state
            )

        test, val = train_test_split(
            test,
            test_size = 0.5,
            stratify = test.iloc[:,-1], # last column is target variable
            random_state = random_state
            )

        datagen = ImageDataGenerator(
                featurewise_center=False,  # set input mean to 0 over the dataset
                samplewise_center=False,  # set each sample mean to 0
                featurewise_std_normalization=False,  # divide inputs by std of the dataset
                samplewise_std_normalization=False,  # divide each input by its std
                zca_whitening=False,  # apply ZCA whitening
                rotation_range=360,  # randomly rotate images in the range (degrees, 0 to 180)
                zoom_range = 0.2, # Randomly zoom image
                width_shift_range=0.2,  # randomly shift images horizontally (fraction of total width)
                height_shift_range=0.2,  # randomly shift images vertically (fraction of total height)
                horizontal_flip=True,  # randomly flip images
                vertical_flip=True, #
                shear_range=0.2,
                brightness_range = datagen_params.get("brightness_range"),
                preprocessing_function = preprocessing_function,
                fill_mode="nearest"
                )
        datagen_iter_train = datagen.flow_from_dataframe(
            train,
            directory   = IMAGE_FILE_ROOT,
            x_col       = x_col,
            y_col       = y_col,
            target_size = target_shape,
            color_mode  = 'rgb',
            class_mode  = 'categorical',
            batch_size  = batch_size,
            shuffle     = True,
            seed        = random_state
        )

        datagen_val = ImageDataGenerator(preprocessing_function = preprocessing_function)

        datagen_iter_val = datagen_val.flow_from_dataframe(
            val,
            directory   = IMAGE_FILE_ROOT,
            x_col       = x_col,
            y_col       = y_col,
            target_size = target_shape,
            color_mode  = 'rgb',
            class_mode  = 'categorical',
            batch_size  = batch_size,
            shuffle     = False,
        )


        datagen_test = ImageDataGenerator(preprocessing_function = preprocessing_function)
        datagen_iter_test = datagen_test.flow_from_dataframe(
            test,
            directory   = IMAGE_FILE_ROOT,
            x_col       = x_col,
            y_col       = y_col,
            target_size = target_shape,
            color_mode  = 'rgb',
            class_mode  = 'categorical',
            batch_size  = batch_size,
            shuffle     = False
        )


        return datagen_iter_train, datagen_iter_test,  datagen_iter_val

# Return a list of all combinations of unique parameters from the given dictionary
def permutate_params(grid_params):

    out = [{}]

    # Loop through each key/val pair
    for param_name, param_list in grid_params.items():

        if len(param_list) == 1:
            for item in out:
                item[param_name] = param_list[0]
        else:
            temp_out = []

            for param_val in param_list:
                for item in out:
                    cloned_item = item.copy()
                    cloned_item[param_name] = param_val
                    temp_out.append(cloned_item)
            out = temp_out
    return out

def gridSearchCNN(
    datagens,
    grid_params,
    file_name,
    optimizer = "adam",
    random_state = 42,
):
    # list of all parameter combinations
    all_params = permutate_params(grid_params)

    # establishing variables
    best_model   = None
    best_score   = 0.0 # no accuracy to start
    best_params  = None
    best_history = None
    test_scores  = None
    train_scores = None
    val_scores  = None


    datagen_iter_train, datagen_iter_val, datagen_iter_test = datagens

    for params in all_params:
        model, history = build_model_from_datagen_new(
            params,
            input_shape        = datagen_iter_train.image_shape,
            datagen_iter_train = datagen_iter_train,
            datagen_iter_val   = datagen_iter_val,
            optimizer          = optimizer,
            file_name          = file_name
        )

        acc = max(history.history["val_categorical_accuracy"])

        # only keeping best
        if acc > best_score:
            print("***Good Accurary found: {:.2%}***".format(acc))
            best_score   = acc
            val_scores  = history.history["val_categorical_accuracy"]
            train_scores = history.history["categorical_accuracy"]
            best_model   = model
            best_params  = params
            best_history = history

    # returns metadata of results (NOTE: retrieving best model from hard disk)
    return {
        "best_model"   : load_model(file_name),
        "best_score"   : best_score,
        "best_params"  : best_params,
        "best_history" : best_history,
        "val_scores"  : val_scores,
        "train_scores" : train_scores
    }

MODEL_PATH = "."
model_name = "original"
stored_model_path = f"{MODEL_PATH}/{model_name}_model.p"

datagen_params = dict()

datagens = create_datagens(
    data,
    datagen_params         = datagen_params,
    batch_size             = 50,
    target_shape           = (IDEAL_WIDTH, IDEAL_HEIGHT),
    preprocessing_function = normalize,
    random_state           = random_state
)

datagen_iter_train, datagen_iter_val, datagen_iter_test = datagens

X_train, y_train = next(datagen_iter_train)
X_val, y_val = next(datagen_iter_val)
X_test, y_test = next(datagen_iter_test)
print(len(X_train))
print(len(X_val))
print(len(X_test))

from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras import layers
from tensorflow.keras import Model
import tensorflow as tf
from tensorflow.keras.optimizers import Adam, RMSprop, SGD

class myCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs={}):
    if(logs.get('acc')>0.999):
      print("\nReached 99.9% accuracy so cancelling training!")
      self.model.stop_training = True

callbacks = myCallback()
from keras.callbacks import ModelCheckpoint, Callback, EarlyStopping, ReduceLROnPlateau
earlystopper = EarlyStopping(monitor='val_loss', patience=20, verbose=1)

# Save the best model during the traning
checkpointer = ModelCheckpoint('./Downloads/Bee_Images_Health_Merge_Class4_Out3/best_model3_cnn.h5'
                                ,monitor='val_acc'
                                ,verbose=1
                                ,save_best_only=True
                                ,save_weights_only=True)

from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras import Sequential, layers, optimizers

n_out = 4

#Load the Base Model inception

image_size = 299
from tensorflow.keras.applications.inception_v3 import InceptionV3
base_model = InceptionV3(input_shape = (image_size, image_size, 3), include_top = False, weights = 'imagenet')

#Compile and Fit

for layer in base_model.layers:
    layer.trainable = False

from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras import layers
from tensorflow.keras import Model
import tensorflow as tf
from tensorflow.keras.optimizers import Adam, RMSprop, SGD

x = layers.Flatten()(base_model.output)
x = layers.Dense(256, activation='relu')(x)
x = layers.Dropout(0.2)(x)

# Add a final sigmoid layer with 1 node for classification output
x = layers.Dense(n_out, activation='softmax')(x)

model = tf.keras.models.Model(base_model.input, x)

model.compile(loss='categorical_crossentropy', optimizer=Adam(lr = 0.0001),metrics=['acc'])

class myCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs={}):
    if(logs.get('acc')>0.999):
      print("\nReached 99.9% accuracy so cancelling training!")
      self.model.stop_training = True

callbacks = myCallback()
from keras.callbacks import ModelCheckpoint, Callback, EarlyStopping, ReduceLROnPlateau
earlystopper = EarlyStopping(monitor='val_loss', patience=20, verbose=1)

# Save the best model during the traning
checkpointer = ModelCheckpoint('./Downloads/Bee_Images_Health_Merge_Class4_Out3/best_inception_model3.h5'
                                ,monitor='val_acc'
                                ,verbose=1
                                ,save_best_only=True
                                ,save_weights_only=True)

# Fit model w/ImageDataGenerator
STEP_SIZE_TRAIN= np.ceil(datagen_iter_train.n/datagen_iter_train.batch_size)
STEP_SIZE_VALID= np.ceil(datagen_iter_val.n/datagen_iter_val.batch_size)

history = model.fit(
            datagen_iter_train,
            validation_data = datagen_iter_val,
            steps_per_epoch = STEP_SIZE_TRAIN,
            #epochs = 100,
            epochs = 20,
            validation_steps = STEP_SIZE_VALID,
            verbose = 2,
            shuffle = True,
            callbacks=[ earlystopper, checkpointer])

model.save('bee-inception.h5')

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(loss))

fig = plt.figure(figsize=(10,6))
plt.plot(epochs,loss,c="red",label="Training")
plt.plot(epochs,val_loss,c="blue",label="Validation")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()

acc = history.history['acc']
val_acc = history.history['val_acc']

epochs = range(len(acc))

fig = plt.figure(figsize=(10,6))
plt.plot(epochs,acc,c="red",label="Training")
plt.plot(epochs,val_acc,c="blue",label="Validation")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()

# Save model and weights
import os
model_name = 'best_model3.h5'

save_dir = os.path.join(os.getcwd(), 'saved_models')


if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
model_path = os.path.join(save_dir, model_name)
#model.save(model_path)
print('Save model and weights at %s ' % model_path)

# Save the model to disk
model_json = model.to_json()
with open("model_json_aug.json", "w") as json_file:
    json_file.write(model_json)

# Model validation
from keras.models import Sequential, Model, model_from_json

json_file = open('model_json_aug.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)

loaded_model.load_weights("best_inception_model3.h5")

print("Loaded model from disk")

from tensorflow.keras.optimizers import Adam, RMSprop, SGD

opt = Adam(lr=0.00001)
import time
start = time.time()
loaded_model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

score = model.predict(datagen_iter_test, batch_size=50)
end = time.time()

import time
start = time.time()
predictions = model.predict_generator(datagen_iter_test)

end = time.time()
print('test time mobilenet', end-start)

preds = []
for pred in predictions:
    preds.append(np.argmax(pred))

print(datagen_iter_test.classes)

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_true = datagen_iter_test.classes,y_pred = np.array(preds))
print(cm)

import csv
y_true = datagen_iter_test.classes
y_pred = np.array(preds)
actualvalues=y_true
predictedvalues=y_pred
filename='Predictions.csv'
with open(filename, "w") as infile:
    writer = csv.writer(infile)
    writer.writerow(["actualvalues", "predictedvalues"])
    for i in zip(actualvalues, predictedvalues):
        writer.writerow(i)

# the confusion matrix heat map plot

import seaborn as sns
import pandas as pd

def print_confusion_matrix(confusion_matrix, class_names, figsize = (10,7), fontsize=14):

    df_cm = pd.DataFrame(
        confusion_matrix, index=class_names, columns=class_names,
    )
    fig = plt.figure(figsize=figsize)
    try:
        heatmap = sns.heatmap(df_cm, annot=True, fmt="d")
    except ValueError:
        raise ValueError("Confusion matrix values must be integers.")

    heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=fontsize)
    heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize=fontsize)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

finaldf = pd.read_csv("Predictions.csv")

finaldf=finaldf.replace(to_replace='0',value="healthy")
finaldf=finaldf.replace(to_replace='2',value="missing queen")
finaldf=finaldf.replace(to_replace='3',value="pesticide")
finaldf=finaldf.replace(to_replace='1',value="ant problems")


classes = finaldf.actualvalues.unique()

classes.sort()
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
# Confusion matrix
c = confusion_matrix(finaldf.actualvalues, finaldf.predictedvalues)
print(accuracy_score(finaldf.actualvalues, finaldf.predictedvalues))
print_confusion_matrix(c, class_names = classes)

# Classification report
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

finaldf['actualvalues']=finaldf['actualvalues'].map(str)
finaldf['predictedvalues']=finaldf['predictedvalues'].map(str)


finaldf=finaldf.replace(to_replace='0',value="healthy")
finaldf=finaldf.replace(to_replace='2',value="missing queen")
finaldf=finaldf.replace(to_replace='3',value="pesticide")
finaldf=finaldf.replace(to_replace='1',value="ant problems")


classes = finaldf.actualvalues.unique()
classes.sort()
print(classification_report(finaldf.actualvalues, finaldf.predictedvalues, target_names=classes, digits=4))

from sklearn.metrics import confusion_matrix
import seaborn as sns

cm = confusion_matrix(finaldf.actualvalues, finaldf.predictedvalues)

cmn = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
fig, ax = plt.subplots(figsize=(10,10))
sns.heatmap(cmn, annot=True, fmt='.2f', xticklabels=classes, yticklabels=classes)
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show(block=False)

#. Mobilenet

from tensorflow import keras
image_size=299

base_model = keras.applications.MobileNet(weights="imagenet",include_top=False,input_shape=(image_size,image_size,3))

base_model.trainable = False
n_out=4
inputs = keras.Input(shape=(image_size,image_size,3))
x = base_model(inputs,training=False)
x = keras.layers.GlobalAveragePooling2D()(x)
x = keras.layers.Dropout(0.2)(x) #tried x = keras.layers.Dropout(0.5)(x)  validation best ~90% https://colab.research.google.com/drive/1SGEC-kefRK40OBDr4h7LZ6D1GQItDXLg#scrollTo=6jxEu-A555us
x = keras.layers.Dense(n_out,activation="softmax")(x)


model = keras.Model(inputs=inputs, outputs=x, name="Bee_MobileNet")

optimizer = keras.optimizers.Adam()
model.compile(optimizer=optimizer,loss=keras.losses.CategoricalCrossentropy(from_logits=True),metrics=[keras.metrics.CategoricalAccuracy()])

from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard,EarlyStopping

earlystopper = EarlyStopping(monitor='loss', patience=20, verbose=1)

checkpointer = ModelCheckpoint('mobile_best_model3.h5'
                                ,monitor='val_categorical_accuracy'
                                ,verbose=1
                                ,save_best_only=True
                                ,save_weights_only=True)


STEP_SIZE_TRAIN= np.ceil(datagen_iter_train.n/datagen_iter_train.batch_size)
STEP_SIZE_VALID= np.ceil(datagen_iter_val.n/datagen_iter_val.batch_size)

history = model.fit_generator(datagen_iter_train,
          validation_data=datagen_iter_val,
          epochs=20,
          steps_per_epoch=STEP_SIZE_TRAIN,
          validation_steps=STEP_SIZE_VALID,
          callbacks=[earlystopper, checkpointer]
         )

model.summary()

X_val, y_val = next(datagen_iter_val)

X_train, y_train = next(datagen_iter_train)
len(X_train)

model.save('bee-mobilenet.h5')

model.summary()

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(loss))

fig = plt.figure(figsize=(10,6))
plt.plot(epochs,loss,c="red",label="Training")
plt.plot(epochs,val_loss,c="blue",label="Validation")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()

acc = history.history['categorical_accuracy']
val_acc = history.history['val_categorical_accuracy']

epochs = range(len(acc))

fig = plt.figure(figsize=(10,6))
plt.plot(epochs,acc,c="red",label="Training")
plt.plot(epochs,val_acc,c="blue",label="Validation")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()

# Save model and weights

import os
model_name = 'bee-mobilenet.h5'

save_dir = os.path.join(os.getcwd(), 'saved_models')

if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
model_path = os.path.join(save_dir, model_name)
print('Save model and weights at %s ' % model_path)

# Save the model to disk
model_json = model.to_json()
with open("model_json_aug.json", "w") as json_file:
    json_file.write(model_json)

# Model validation

from keras.models import Sequential, Model, model_from_json

json_file = open('model_json_aug.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)

loaded_model.load_weights('bee-mobilenet.h5')

print("Loaded model from disk")

from tensorflow.keras.optimizers import Adam, RMSprop, SGD

opt = Adam(lr=0.0001)
loaded_model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

score = model.predict(datagen_iter_test, batch_size=50)

predictions = model.predict_generator(datagen_iter_test)

preds = []
for pred in predictions:
    preds.append(np.argmax(pred))

print(datagen_iter_test.classes)

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_true = datagen_iter_test.classes,y_pred = np.array(preds))
print(cm)

print(y_true)
print(y_pred)

import csv
y_true = datagen_iter_test.classes
y_pred = np.array(preds)
actualvalues=y_true
predictedvalues=y_pred
filename='Predictions.csv'
with open(filename, "w") as infile:
    writer = csv.writer(infile)
    writer.writerow(["actualvalues", "predictedvalues"])
    for i in zip(actualvalues, predictedvalues):
        writer.writerow(i)

# Confusion matrix heat map plot

import seaborn as sns
import pandas as pd

def print_confusion_matrix(confusion_matrix, class_names, figsize = (10,7), fontsize=14):
    df_cm = pd.DataFrame(
        confusion_matrix, index=class_names, columns=class_names,
    )
    fig = plt.figure(figsize=figsize)
    try:
        heatmap = sns.heatmap(df_cm, annot=True, fmt="d")
    except ValueError:
        raise ValueError("Confusion matrix values must be integers.")

    heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=fontsize)
    heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize=fontsize)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


finaldf = pd.read_csv("Predictions.csv")

finaldf=finaldf.replace(to_replace='0',value="healthy")
finaldf=finaldf.replace(to_replace='2',value="missing queen")
finaldf=finaldf.replace(to_replace='3',value="pesticide")
finaldf=finaldf.replace(to_replace='1',value="ant problems")

classes = finaldf.actualvalues.unique()

classes.sort()
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

c = confusion_matrix(finaldf.actualvalues, finaldf.predictedvalues)
print(accuracy_score(finaldf.actualvalues, finaldf.predictedvalues))
print_confusion_matrix(c, class_names = classes)

# Classification report
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

mapping = {0 : 'cooling', 1:'pollen',2:'Varroa',3:'wasps'}

finaldf['actualvalues']=finaldf['actualvalues'].map(str)
finaldf['predictedvalues']=finaldf['predictedvalues'].map(str)

finaldf=finaldf.replace(to_replace='0',value="healthy")
finaldf=finaldf.replace(to_replace='2',value="missing queen")
finaldf=finaldf.replace(to_replace='3',value="pesticide")
finaldf=finaldf.replace(to_replace='1',value="ant problems")

classes = finaldf.actualvalues.unique()
classes.sort()
print(classification_report(finaldf.actualvalues, finaldf.predictedvalues, target_names=classes, digits=4))