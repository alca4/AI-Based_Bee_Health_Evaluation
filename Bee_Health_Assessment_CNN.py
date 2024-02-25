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

# Return the count of unique values
raw_data['health'].value_counts()

raw_data.shape

health_counts = raw_data["health"].value_counts()

plt.title("Counts of bee $health$ categories")
g = sns.barplot(x = health_counts, y = health_counts.index);
g.set_xlabel("Frequency");

#Clearing unnecessary columns
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
show_image(raw_images[0])

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

# Return the average from the given distribution above the cutoff values

def get_best_average(dist, cutoff = .5):


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

from numpy.lib.function_base import average
from pandas.core.algorithms import mode
# from pandas.core.algorithms import mean
min(wh[0]), max(wh[0]), average(wh[0]), mode(wh[0])

from pandas.core.algorithms import mode
from numpy.lib.function_base import average
min(wh[1]), max(wh[1]), average(wh[1]),  mode(wh[1])

import numpy as np
from skimage.transform import rescale, resize, rotate
from skimage.color import rgb2gray

class ImageHandler():
    def __init__(self, images):
        self.images = np.asanyarray([np.asarray(image) for image in images])
        self.index = np.array(range(len(self.images)))
        self._is_clone = False

    @property
    def images_for_display(self):
        '''getter for image data intended for visualization (no negatives, and scaled appropriately)'''

        if len(self.images.shape) == 4 and self.images.shape[3] == 1: # grayscale can simply return as is without 3rd dim.
            return np.squeeze(self.images)

        if len(self.images.shape) == 1: # unaltered .images
            return self.images

        return self._handle_normalized_state(
            lambda : self.images / 255,
            lambda : self.images + .5,
            lambda err : "Could not convert images to standard format"
        )

    def _handle_normalized_state(self, if_255_func, if_norm_func, err_func = None):
        try:
            return if_255_func() if self.images.max() > 1 else if_norm_func()
        except Exception as e:
            if err_func:
                raise Exception(err_func(e))
            raise e

    def get_by_index(self, indeces):
        # should return a new handler for chaining
        return ImageHandler(self.images[indeces])

    def resize(self, resizing):
        '''
        resizing = (W,H)
        '''
        context = self._get_context()
        context.images = np.asanyarray([resize(image, resizing, anti_aliasing=True, mode = "constant", preserve_range = True) for image in context.images])
        return context

    def grayscale(self):
        context = self._get_context()
        context.images = np.expand_dims(rgb2gray(context.images), axis = 3)
        return context

    def rotate(self):
        context = self._get_context()
        out = []
        # original
        out.extend(context.images)

        #rotated
        rotated = [rotate(image, angle) for angle in range(90,360,90) for image in context.images]
        out.extend(rotated)

        context.images = np.asanyarray(out)

        ## alter context index
        original      = context.index
        angles        = np.repeat(context.index, 3)
        all_angles    = np.append(original, angles)
        context.index = all_angles
        return context

    def invert(self):
        context = self._get_context()

        context.images =  context._handle_normalized_state(
            lambda : 255 - context.images,
            lambda : .5 - context.images
        )
        return context

    def normalize(self):
        if self.images.dtype != "float64":
            raise Exception("Images are differently shaped. Try to .resize() first.")
        context = self._get_context()
        context.images =  context._handle_normalized_state(
            lambda : (context.images / 255) - 0.5,
            lambda : context.images - 0.5
        )
        return context

    def transform(self, resize = False, normalize = False, grayscale = False, invert = False, rotate = False):
        context = self._get_context()
        if resize:
            context.resize(resize)
        if normalize:
            context.normalize()
        if grayscale:
            context.grayscale()
        if rotate:
            context.rotate()
        if invert:
            context.invert()
        return context

    def _clone(self):
        clone = ImageHandler(self.images.copy())
        clone._is_clone = True
        return clone

    def _get_context(self):
        '''returns context. if current object not clone, will return a cloned verson of self'''
        return self._clone() if not self._is_clone else self

IDEAL_WIDTH, IDEAL_HEIGHT = 150, 150
IDEAL_WIDTH, IDEAL_HEIGHT

#Visualizing resized images
resizing = (IDEAL_WIDTH, IDEAL_HEIGHT, 3)

sample_img_store = ImageHandler(raw_images[:201]).resize(resizing)

old_image_1 = raw_images[100]
new_image_1 = sample_img_store.images_for_display[100]

old_image_2 = raw_images[200]
new_image_2 = sample_img_store.images_for_display[200]
show_images(
    [old_image_1, new_image_1, old_image_2, new_image_2],
    titles = ["Original #1 (tall)", "Resized #1", "Original #2 (wide)", "Resized #2"],
    show_size = True
)

data["health"].value_counts(normalize = True)

def normalize(image):
    return (image/255. - 0.5)

def create_datagens(    data, datagen_params,
    target_shape, batch_size, x_col="file", y_col="health", IMAGE_FILE_ROOT = './Downloads/Bee_Images_Health_Merge_Class4_Out3/',
    random_state = 42, preprocessing_function = None):

        data[y_col] = data[y_col].astype(str) # coercion needed for datagen
        # train/test split
        train, test = train_test_split(
            data,
            test_size = 0.2,
            stratify = data.iloc[:,-1], # assumed last column is target variable
            random_state = random_state
            )

        test, val = train_test_split(
            test,
            test_size = 0.5,
            stratify = test.iloc[:,-1], # assumed last column is target variable
            random_state = random_state
            )

        # training ImageDataGenerator

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
                vertical_flip=True,
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

        #validation ImageDataGenerator
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


        # testing ImageDataGenerator
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

def permutate_params(grid_params):
    '''Returns a list of all combinations of unique parameters from the given dictionary'''
    out = [{}]

    # loop through each key/val pair
    for param_name, param_list in grid_params.items():
        # shortcircut - no need to permute single items
        if len(param_list) == 1:
            for item in out:
                item[param_name] = param_list[0]
        else:
            temp_out = []
            # for each item in the param, clone entire growing list and add param to each
            for param_val in param_list:
                for item in out:
                    cloned_item = item.copy()
                    cloned_item[param_name] = param_val
                    temp_out.append(cloned_item)
            out = temp_out
    return out

def build_model_from_datagen_new(
    params = dict(),
    input_shape = (),
    datagen_iter_train = None,
    datagen_iter_val = None,
    optimizer = "adam",
    file_name = None
):
    '''Returns a fitted convolutional neural network with the given parameters and data.'''
    kernel_size = 3
    dropout = .5
    activation_func = "relu"

    conv__filters_1 = params.get('conv__filters_1') or 32
    conv__filters_2 = params.get('conv__filters_2') or 16
    conv__filters_3 = params.get('conv__filters_3') or 32
    conv__filters_4 = params.get('conv__filters_3') or 16
    density_units_1 = params.get('density_units_1') or 32
    density_units_2 = params.get('density_units_2') or 32

    epochs = 20
    epochs = epochs

    model = Sequential()
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same', input_shape=(IDEAL_WIDTH, IDEAL_HEIGHT,3)))
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(1024, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(4, activation='softmax'))

    # compiling model
    model.compile(
        loss      = 'categorical_crossentropy',
        optimizer = optimizer,
        metrics   = ['categorical_accuracy']
    )

    # fitting model w/ImageDataGenerator
    STEP_SIZE_TRAIN= np.ceil(datagen_iter_train.n/datagen_iter_train.batch_size)
    STEP_SIZE_VALID= np.ceil(datagen_iter_val.n/datagen_iter_val.batch_size)

    history = model.fit_generator(
        generator           = datagen_iter_train,
        steps_per_epoch     = STEP_SIZE_TRAIN,
        validation_data     = datagen_iter_val,
        validation_steps    = STEP_SIZE_VALID,
        epochs              = epochs,
        callbacks           = [callbacks.ModelCheckpoint(file_name, save_best_only=True, mode='auto', period=1)]
    )

    return (model, history)

def gridSearchCNN(
    datagens,
    grid_params,
    file_name,
    optimizer = "adam",
    random_state = 42,
):
    all_params = permutate_params(grid_params)

    # establishing variables
    best_model   = None
    best_score   = 0.0 # no accuracy to start
    best_params  = None
    best_history = None
    test_scores  = None
    train_scores = None
    val_scores  = None

    #datagen_iter_train, datagen_iter_test = datagens
    datagen_iter_train, datagen_iter_val, datagen_iter_test = datagens

    # for each permuted parameter, try fitting a model (NOTE: the best model is saved to disk with file_name)
    for params in all_params:
        model, history = build_model_from_datagen_new(
            params,
            input_shape        = datagen_iter_train.image_shape,
            datagen_iter_train = datagen_iter_train,
            datagen_iter_val   = datagen_iter_val,
            #datagen_iter_test   = datagen_iter_test,
            optimizer          = optimizer,
            file_name          = file_name
        )

        acc = max(history.history["val_categorical_accuracy"])

        # only keeping best
        if acc > best_score:
            print("***Good Accurary found: {:.2%}***".format(acc))
            best_score   = acc
            #test_scores  = history.history["test_categorical_accuracy"]
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

def conf_matrix_stats(y_test, preds):
    ''' Return key confusion matrix metrics given true and predicted values'''
    cm = confusion_matrix(y_test, preds)
    TP, FP, FN, TN, = cm[1,1], cm[0,1], cm[1,0], cm[0,0]
    total = (TP + FP + FN + TN)
    acc = (TP + TN ) / total
    miss = 1 - acc
    sens = TP / (TP + FN)
    spec = TN / (TN + FP)
    prec = TP / (TP + FP)
    return {"accuracy": acc, "miss_rate": miss, "sensitivity": sens, "specification": spec, "precision": prec}

def graph_loss(history):

    # Check out our train loss and test loss over epochs.
    train_loss = history.history['loss']
    val_loss = history.history['val_loss']
    xticks = np.array(range(len(train_loss)))
    # Set figure size.
    plt.figure(figsize=(12, 8))

    # Generate line plot of training, testing loss over epochs.
    plt.plot(train_loss, label='Training Loss', color='#185fad')
    plt.plot(val_loss, label='Testing Loss', color='orange')

    # Set title
    plt.title('Training and validating Loss by Epoch', fontsize = 25)
    plt.xlabel('Epoch', fontsize = 18)
    plt.ylabel('categorical Crossentropy', fontsize = 18)
    plt.xticks(xticks[::5], (xticks+1)[::5])

    plt.legend(fontsize = 18);

def show_results(images, shapes, title, ncols = 4, height = 2, width = 2):
    '''Plots list of given images'''
    nrows = int(np.ceil(len(images)/ncols))
    f, ax = plt.subplots(nrows=nrows,ncols=ncols, figsize=(ncols * width, nrows * height))
    ax = ax.flatten() if type(ax) == np.ndarray else ax
    i = None
    for i, image in enumerate(images):
        _title = f"Orig. size: {shapes[i][0]}x{shapes[i][1]}\n{title} #{i+1}"
        show_image(image, ax = ax[i] if type(ax) == np.ndarray else ax, title = _title)

    # removing extraneous subplots
    while i and type(ax) == np.ndarray and i < len(ax) - 1:
        i += 1
        f.delaxes(ax[i])

MODEL_PATH = "./Downloads"
model_name = "original"
stored_model_path = f"{MODEL_PATH}/{model_name}_model.pkl"

datagen_params = dict()

datagens = create_datagens(
    data,
    datagen_params         = datagen_params,
    batch_size             = 50, #hyperparameter
    target_shape           = (IDEAL_WIDTH, IDEAL_HEIGHT),
    preprocessing_function = normalize,
    random_state           = random_state
)

grid_params = {
    "conv__filters_1" : [32],
    "conv__filters_2" : [48],
    "conv__filters_3" : [64],
    "conv__filters_4" : [128],
    "density_units_1" : [256],
    "density_units_2" : [64],
    "batch_size"      : [64],
    "epochs"          : [50]
}

best_original_model = gridSearchCNN(
    datagens     = datagens,
    grid_params  = grid_params,
    random_state = random_state,
    optimizer    = 'Adam',
    file_name    = f"./best_{model_name}.h5"
)

pickle.dump(best_original_model, open(stored_model_path, 'wb')) #saving metadata

import pickle
best_original_model = pickle.load(open(stored_model_path, 'rb')) #loading metadata

best_original_model["best_model"].summary()

graph_loss(best_original_model["best_history"])

#Evaluation

history_dict = best_original_model["best_history"].history
loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']
epochs = range(1, len(loss_values)+1)

plt.figure(figsize=(8,6))
plt.plot(epochs, loss_values, 'bo', label='Training loss')
plt.plot(epochs, val_loss_values, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

train_test_scores = list(zip(best_original_model["train_scores"], best_original_model["val_scores"]))
train_test_scores.sort(key=lambda scores: scores[1])
train_test_scores[-1]

# Using testing set to assess accuracy
datagen_iter_test = datagens[2]
datagen_iter_test.reset()

predictions = best_original_model["best_model"].predict_generator(datagen_iter_test, steps = datagen_iter_test.n)
print('predictions', len(predictions))

preds = []
for pred in predictions:
    preds.append(np.argmax(pred))

y_true = datagen_iter_test.classes
print('y_true', len(y_true))

from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np

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
    writer.writerow(["actualvalues", "predictedvalues"])    #Write Header
    for i in zip(actualvalues, predictedvalues):
        writer.writerow(i)                 #Write Content

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
finaldf=finaldf.replace(to_replace='1',value="ant problem")

classes = finaldf.actualvalues.unique()
classes.sort()
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
c = confusion_matrix(finaldf.actualvalues, finaldf.predictedvalues)
print(accuracy_score(finaldf.actualvalues, finaldf.predictedvalues))
print_confusion_matrix(c, class_names = classes)

# Classification report
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

mapping = {0 : 'healthy', 3:'pesticide',2:'missing queen', 1:'ants problems'}

finaldf['actualvalues']=finaldf['actualvalues'].map(str)
finaldf['predictedvalues']=finaldf['predictedvalues'].map(str)

finaldf=finaldf.replace(to_replace='0',value="healthy")
finaldf=finaldf.replace(to_replace='2',value="missing queen")
finaldf=finaldf.replace(to_replace='3',value="pesticide")
finaldf=finaldf.replace(to_replace='1',value="ant problem")

classes = finaldf.actualvalues.unique()
classes.sort()
print(classification_report(finaldf.actualvalues, finaldf.predictedvalues, target_names=classes, digits=4))

from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np

classes = ['healthy', 'ant problem','missing queen','pesticide']

cm = confusion_matrix(y_true = y_true,y_pred = preds)
print(cm)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)

disp.plot(cmap=plt.cm.Blues)
plt.show()

from sklearn.metrics import confusion_matrix
import seaborn as sns

cm = confusion_matrix(y_true, preds)
# Normalise
cmn = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
fig, ax = plt.subplots(figsize=(10,10))
sns.heatmap(cmn, annot=True, fmt='.0%', xticklabels=classes, yticklabels=classes)
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show(block=False)