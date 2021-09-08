### TRAINING ###
# This is where the majority of your code should live for training the model. #

import argparse, logging, sys, os, tarfile, io, math, glob
import numpy as np, pandas as pd

from os import getenv
from os.path import abspath, basename, split,dirname
from shutil import copyfile

import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras.applications import EfficientNetB0, EfficientNetB2, EfficientNetB3, EfficientNetB5
from tensorflow.keras import layers
from tensorflow import keras
from tensorflow.keras.applications.efficientnet import preprocess_input
import tensorflow.keras.backend as K

import efficientnet.tfkeras as efn
from tensorflow.keras import mixed_precision
from tensorflow.keras.preprocessing import image
from PIL import Image
from tqdm.auto import tqdm

import albumentations as A
from sklearn.metrics import accuracy_score

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
CHANNELS = 3
IMG_SIZE = 224
NUM_CLASSES = 28
BATCH_SIZE = 32
LBL = dict(zip(['B_BSMUT1', 'B_CLEV5B', 'B_DISTO', 'B_GRMEND', 'B_HDBARL',
       'B_PICKLD', 'B_SKINED', 'B_SOUND', 'B_SPRTED', 'B_SPTMLD',
       'O_GROAT', 'O_HDOATS', 'O_SEPAFF', 'O_SOUND', 'O_SPOTMA',
       'WD_RADPODS', 'WD_RYEGRASS', 'WD_SPEARGRASS', 'WD_WILDOATS',
       'W_DISTO', 'W_FLDFUN', 'W_INSDA2', 'W_PICKLE', 'W_SEVERE',
       'W_SOUND', 'W_SPROUT', 'W_STAIND', 'W_WHITEG'], range(28)))
cls_map = dict(zip(LBL.values(),LBL.keys()))

model_version = '001'

# Work around for a SageMaker path issue
# (see https://github.com/aws/sagemaker-python-sdk/issues/648)
# WARNING - removing this may cause the submission process to fail
if abspath("/opt/ml/code") not in sys.path:
    sys.path.append(abspath("/opt/ml/code"))

# ENABLE MIXED PRECISION for speed
tf.config.optimizer.set_experimental_options({"auto_mixed_precision": True})
print('Mixed precision enabled')


# Extract train and val data to respctive folders
os.system("mkdir /opt/ml/input/data/training/train")
os.system("mkdir /opt/ml/input/data/training/val")

data_dir = getenv("SM_CHANNEL_TRAINING", "/opt/ml/input/data/training")
train_unzip_cmd = "unzip -q /opt/ml/input/data/training/train.zip -d /opt/ml/input/data/training/train"
valid_unzip_cmd = "unzip -q /opt/ml/input/data/training/val.zip -d /opt/ml/input/data/training/val"
os.system(train_unzip_cmd)
os.system(valid_unzip_cmd)

# For speedup training we load all train and valid images into RAM

X_train = []
y_train = []

for fn in tqdm(glob.iglob('/opt/ml/input/data/training/train/**/*.png', recursive=True)):
    file_name = os.path.basename(fn)
    path = os.path.abspath(fn)
    folder = fn.split('/')[-2] #os.path.split(os.path.dirname(path))[1]
    if len(file_name.split("-")) > 2:  # ignore master image with may grains, raw image names are in guid format
        
        img = Image.open(fn)
        img = img.convert('RGB')
        img = img.resize((IMG_SIZE, IMG_SIZE), Image.NEAREST)
        img_array = image.img_to_array(img)
        img_array = img_array.astype(np.uint8)

        X_train.append(img_array)
        y_train.append(LBL[folder])

X_train = np.array(X_train)
y_train = np.array(y_train)

X_valid = []
y_valid = []

for fn in tqdm(glob.iglob('/opt/ml/input/data/training/val/**/*.png', recursive=True)):
    file_name = os.path.basename(fn)
    path = os.path.abspath(fn)
    folder = fn.split('/')[-2] #os.path.split(os.path.dirname(path))[1]
    if len(file_name.split("-")) > 2:  # ignore master image with may grains, raw image names are in guid format

        img = Image.open(fn)
        img = img.convert('RGB')
        img = img.resize((IMG_SIZE, IMG_SIZE), Image.NEAREST)
        img_array = image.img_to_array(img)
        img_array = img_array.astype(np.uint8)

        X_valid.append(img_array)
        y_valid.append(LBL[folder])

X_valid = np.array(X_valid)
y_valid = np.array(y_valid)

logger.info(f" IMAGE SIZES {X_train.shape}, {y_train.shape} , {X_valid.shape}, {y_valid.shape}")


# We are using tensorflow data generators to speed up and using albumentation augmentations

class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, X, y, list_IDs, batch_size=BATCH_SIZE, shuffle=False, augment=False, mixup=False, cutmix=False, labels=True, onehot=True):
        self.X = X
        self.y = y
        self.augment = augment
        self.mixup = mixup
        self.cutmix = cutmix
        self.list_IDs = list_IDs
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.labels = labels
        self.onehot = onehot
        self.on_epoch_end()
    
    def __len__(self):
        'Denotes the number of batches per epoch'
        ct = len(self.list_IDs) // self.batch_size
        ct += int((len(self.list_IDs) % self.batch_size)!=0)
        return ct
    
    def __getitem__(self, index):
        'Generate one batch of data'
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        X, y = self.__data_generation(indexes)
        if self.augment: X = self.__augment_batch(X)
        if self.mixup: X, y = self.__mix_up(X, y)
        if self.cutmix: X, y = self.__cut_mix(X, y)
        if self.labels: return X, y
        else: return X
    
    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange( len(self.list_IDs) )
        if self.shuffle: np.random.shuffle(self.indexes)
    
    def __data_generation(self, indexes):
        'Generates data containing batch_size samples'        
        X = self.X[ self.list_IDs[indexes], ]
        if self.onehot:
            y = np.zeros((len(indexes),NUM_CLASSES))
            for j in range(len(indexes)):
                y[j, int(self.y[self.list_IDs[indexes[j]]])] = 1
        else:
            y = self.y[ self.list_IDs[indexes]]

        return X, y
    
    def __random_transform(self, img):
        composition = A.Compose([
            A.Transpose(p=0.2),
            A.VerticalFlip(p=0.2),
            A.HorizontalFlip(p=0.2),
            A.RandomBrightness(limit=0.2, p=0.05),
            A.RandomContrast(limit=0.2, p=0.05),
            A.OneOf([
                A.ShiftScaleRotate(rotate_limit=8,scale_limit=0.16,shift_limit=0,border_mode=0,value=0,p=0.5),
                A.CoarseDropout(max_holes=16,max_height=IMG_SIZE//10,max_width=IMG_SIZE//10,fill_value=0,p=0.5),
                A.Cutout(max_h_size=int(IMG_SIZE * 0.375), max_w_size=int(IMG_SIZE * 0.375), num_holes=1, p=0.5)
            ], p=0.5),
            A.ShiftScaleRotate(rotate_limit=0, scale_limit=0., shift_limit=0.12, border_mode=0, value=0, p=0.5)
        ])

        return composition(image=img)['image']
    

    def __augment_batch(self, img_batch):
        for i in range(img_batch.shape[0]):
            img_batch[i, ] = self.__random_transform(img_batch[i, ])
        return img_batch
    
    def __mix_up(self, x, y):
        lam = np.random.beta(1.0, 1.0) #np.random.beta(0.2, 0.4)
        ori_index = np.arange(int(len(x)))
        index_array = np.arange(int(len(x)))
        np.random.shuffle(index_array)        
        
        mixed_x = lam * x[ori_index] + (1 - lam) * x[index_array]
        mixed_y = lam * y[ori_index] + (1 - lam) * y[index_array]
        mixed_x = np.clip(mixed_x, 0, 255).astype(np.uint8)

        return mixed_x, mixed_y
    
    def __cut_mix(self, x, y):

        ori_index = np.arange(int(len(x)))
        index_array = np.arange(int(len(x)))
        np.random.shuffle(index_array)

        lam = np.random.beta(1.0, 1.0)#np.random.beta(0.2, 0.4)

        bbx1, bby1, bbx2, bby2 = rand_bbox(lam)
        x[:, bbx1:bbx2, bby1:bby2, :] = x[index_array, bbx1:bbx2, bby1:bby2, :]
        # adjust lambda to exactly match pixel ratio
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (IMG_SIZE * IMG_SIZE))

        mixed_y = lam * y[ori_index] + (1 - lam) * y[index_array]
        #y = [targets1, shuffled_targets1, targets2, shuffled_targets2, targets3, shuffled_targets3, lam]
        return x, mixed_y

def rand_bbox(lam):
    W = IMG_SIZE
    H = IMG_SIZE
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


def train(args):
    global NUM_CLASSES
    """Train
    """
    logger.info("calling training function")

    logger.info("Training Model")
    
    train_gen = DataGenerator(X_train,y_train,np.arange(len(X_train)),shuffle=True,augment=False, batch_size=BATCH_SIZE, onehot=True)
    valid_gen = DataGenerator(X_valid,y_valid,np.arange(len(X_valid)),shuffle=False,augment=False, batch_size=BATCH_SIZE, onehot=True)
    train_gen_aug = DataGenerator(X_train,y_train,np.arange(len(X_train)),shuffle=True,augment=True, batch_size=BATCH_SIZE, onehot=True)
    
    # sample weights to each sample for calculating weighted accuracy metrics
    w_valid = []
    for l in y_valid:
        if l in [7, 13, 24]:
            w_valid.append(8.94)
        else:
            w_valid.append(1.0)
    
    ############################################# MODEL 1 ########################################

    print("#### MODEL 1 - E0 - AUGMENT ONLY")
    save_path1 = "best1.h5"
    checkpoint1 = tf.keras.callbacks.ModelCheckpoint(save_path1, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max', save_weights_only=True)
    early_stop1 = keras.callbacks.EarlyStopping(monitor='val_accuracy', mode='max', verbose=1, patience=10, restore_best_weights=False)
    lr_reducer1 = tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", patience=2, min_lr=1e-6)

    model1 = build_model0(NUM_CLASSES)
    model1.fit(train_gen_aug, validation_data=valid_gen, epochs = 17, verbose=2,  callbacks=[lr_reducer1, early_stop1, checkpoint1])
    model1.load_weights(save_path1)
    model1.save(save_path1)

    # finding best weights 
    y_pred1 = model1.predict(X_valid, batch_size=BATCH_SIZE)
    W1 = find_best_w(y_pred1, y_valid, w_valid)
    ny_pred = np.multiply(y_pred1, W1)
    score = accuracy_score(y_true=y_valid, y_pred=ny_pred.argmax(axis=1), sample_weight=w_valid)

    print("MODEL 1 BEST SCORE : ", score)
    W1 = tf.constant(W1, dtype=tf.float32)

    ############################################# MODEL 2 ########################################

    print("#### MODEL 2 - E1 - AUGMENT ONLY")
    save_path2 = "best2.h5"
    checkpoint2 = tf.keras.callbacks.ModelCheckpoint(save_path2, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max', save_weights_only=True)
    early_stop2 = keras.callbacks.EarlyStopping(monitor='val_accuracy', mode='max', verbose=1, patience=10, restore_best_weights=False)

    lr_reducer2 = tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", patience=2, min_lr=1e-6)

    model2 = build_model1(NUM_CLASSES)
    model2.fit(train_gen_aug, validation_data=valid_gen, epochs = 15, verbose=2,  callbacks=[lr_reducer2, early_stop2, checkpoint2])
    model2.load_weights(save_path2)
    model2.save(save_path2)

    # finding best weights 
    y_pred2 = model2.predict(X_valid, batch_size=BATCH_SIZE)
    W2 = find_best_w(y_pred2, y_valid, w_valid)
    ny_pred = np.multiply(y_pred2, W2)
    score = accuracy_score(y_true=y_valid, y_pred=ny_pred.argmax(axis=1), sample_weight=w_valid)

    print("MODEL 2 BEST SCORE : ", score)
    W2 = tf.constant(W2, dtype=tf.float32)

    ############################################# MODEL 3 ########################################

    print("#### MODEL 3 - E2 - AUGMENT ONLY")
    save_path3 = "best3.h5"
    checkpoint3 = tf.keras.callbacks.ModelCheckpoint(save_path3, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max', save_weights_only=True)
    early_stop3 = keras.callbacks.EarlyStopping(monitor='val_accuracy', mode='max', verbose=1, patience=10, restore_best_weights=False)

    lr_reducer3 = tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", patience=2, min_lr=1e-6)

    model3 = build_model2(NUM_CLASSES)
    model3.fit(train_gen_aug, validation_data=valid_gen, epochs = 15, verbose=2,  callbacks=[lr_reducer3, early_stop3, checkpoint3])
    model3.load_weights(save_path3)
    model3.save(save_path3)

    # finding best weights 
    y_pred3 = model3.predict(X_valid, batch_size=BATCH_SIZE)
    W3 = find_best_w(y_pred3, y_valid, w_valid)
    ny_pred = np.multiply(y_pred3, W3)
    score = accuracy_score(y_true=y_valid, y_pred=ny_pred.argmax(axis=1), sample_weight=w_valid)

    print("MODEL 3 BEST SCORE : ", score)
    W3 = tf.constant(W3, dtype=tf.float32)

    ##### Ensamble Weight ######
    # finding best weights for ensable predictions
    y_pred = (y_pred1  + y_pred2 + y_pred3)
    EW = find_best_w(y_pred, y_valid, w_valid)
    ny_pred = np.multiply(y_pred, EW)
    score = accuracy_score(y_true=y_valid, y_pred=ny_pred.argmax(axis=1), sample_weight=w_valid)

    print("BEST ENSAMBLE SCORE : ", score)
    EW = tf.constant(EW, dtype=tf.float32)

    
    def ensamble():

        inputs = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
        model1 = tf.keras.models.load_model(save_path1, custom_objects={"WeightedAccuracy": WeightedAccuracy})
        model2 = tf.keras.models.load_model(save_path2, custom_objects={"WeightedAccuracy": WeightedAccuracy})
        model3 = tf.keras.models.load_model(save_path3, custom_objects={"WeightedAccuracy": WeightedAccuracy})

        y1 = model1(inputs)
        
        y2 = model2(inputs)

        y3 = model3(inputs)

        outputs = y1 + y2 + y3
        outputs = tf.multiply(outputs, EW)
        return keras.Model(inputs, outputs)
    
    model = ensamble()
    y_pred = model.predict(X_valid, batch_size=BATCH_SIZE)
    
    score = accuracy_score(y_true=y_valid, y_pred=y_pred.argmax(axis=1), sample_weight=w_valid)
    print("## Ensabmle score : ", score)

    save_model(model,args.model_dir)


def find_best_w(y_pred, y_valid, w_valid):
    """Finding best weights for y_pred and y_valid that maximise compitation metric 
    We are finding weights for only high weighted class those are [7, 13, 24]
    
    Args:
        y_pred ([list]): model predictions
        y_valid ([list]): true values
        w_valid ([list]): sample weights for each true value

    Returns:
        [list]: weights for each class
    """
    FINAL_W = []
    FINAL_S = []
    W = np.array([1, 1, 1, 1, 1, 1, 1, 8.94*1, 1, 1, 1, 1, 1, 8.94*1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 8.94*1, 1, 1, 1]).astype(np.float32)
    for c in [7, 13, 24]:
        best_score = 0
        best_w = 0
        w = np.array([1, 1, 1, 1, 1, 1, 1, 8.94*1, 1, 1, 1, 1, 1, 8.94*1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 8.94*1, 1, 1, 1]).astype(np.float32)
        for i in np.arange(0, 12, 0.01):
            w[c] = 8.94 * i
            wy_pred = np.multiply(y_pred, w)
            s = accuracy_score(y_true=y_valid, y_pred=wy_pred.argmax(axis=1), sample_weight=w_valid)
            if s > best_score:
                best_score = s
                best_w = i
        FINAL_W.append(best_w)
        FINAL_S.append(best_score)

    for i,c in enumerate([7, 13, 24]):
        W[c] = W[c] * FINAL_W[i]
    
    return W


def build_model0(num_classes):
    inp = tf.keras.layers.Input(shape=(IMG_SIZE,IMG_SIZE,3))
    scaled_input = layers.experimental.preprocessing.Rescaling(1./255,name="rescaling")(inp)
    base = efn.EfficientNetB0(input_shape=(IMG_SIZE,IMG_SIZE,3),include_top=False, weights='noisy-student')
    x = base(scaled_input)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    top_dropout_rate = 0.2
    x = layers.Dropout(top_dropout_rate, name="top_dropout")(x)
    x = layers.Dense(NUM_CLASSES, name="pred")(x)
    x = layers.Activation('softmax', dtype='float32', name="predictions")(x)
    model = tf.keras.Model(inputs=inp,outputs=x)

    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

    model.compile(
        optimizer=optimizer, 
        loss="categorical_crossentropy", 
        metrics=['accuracy', WeightedAccuracy()],
    )
    return model

def build_model1(num_classes):
    inp = tf.keras.layers.Input(shape=(IMG_SIZE,IMG_SIZE,3))
    scaled_input = layers.experimental.preprocessing.Rescaling(1./255,name="rescaling")(inp)
    base = efn.EfficientNetB1(input_shape=(IMG_SIZE,IMG_SIZE,3),include_top=False, weights='noisy-student')
    x = base(scaled_input)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    top_dropout_rate = 0.2
    x = layers.Dropout(top_dropout_rate, name="top_dropout")(x)
    x = layers.Dense(NUM_CLASSES, name="pred")(x)
    x = layers.Activation('softmax', dtype='float32', name="predictions")(x)
    model = tf.keras.Model(inputs=inp,outputs=x)

    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

    model.compile(
        optimizer=optimizer, 
        loss="categorical_crossentropy",
        metrics=['accuracy', WeightedAccuracy()],
    )
    return model

def build_model2(num_classes):
    inp = tf.keras.layers.Input(shape=(IMG_SIZE,IMG_SIZE,3))
    scaled_input = layers.experimental.preprocessing.Rescaling(1./255,name="rescaling")(inp)
    base = efn.EfficientNetB2(input_shape=(IMG_SIZE,IMG_SIZE,3),include_top=False, weights='noisy-student')
    x = base(scaled_input)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    top_dropout_rate = 0.2
    x = layers.Dropout(top_dropout_rate, name="top_dropout")(x)
    x = layers.Dense(NUM_CLASSES, name="pred")(x)
    x = layers.Activation('softmax', dtype='float32', name="predictions")(x)
    model = tf.keras.Model(inputs=inp,outputs=x)

    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

    model.compile(
        optimizer=optimizer, 
        loss="categorical_crossentropy",
        metrics=['accuracy', WeightedAccuracy()],
    )
    return model

class WeightedAccuracy(tf.keras.metrics.Metric):
    def __init__(self, name="Waccuracy", **kwargs):
        super(WeightedAccuracy, self).__init__(name=name, **kwargs)
        self.true_positives = self.add_weight(name="ctp", initializer="zeros")
        self.m = tf.keras.metrics.Accuracy()
        
    def update_state(self, y_true, y_pred, sample_weight=None):

        y_true = tf.argmax(y_true, axis=1)
        y_pred = tf.argmax(y_pred, axis=1)

        sample_w1 = tf.where(y_true == 7, x=8.94, y=0.0)
        sample_w2 = tf.where(y_true == 13, x=8.94, y=0.0)
        sample_w3 = tf.where(y_true == 24, x=8.94, y=0.0)
        sample_w = sample_w1 + sample_w2 + sample_w3
        sample_ww = tf.where(sample_w == 0, x=1.0, y=8.94)
        
        self.m.update_state(y_true, y_pred, sample_weight=sample_ww)

    def result(self):
        return self.m.result()

    def reset_state(self):
        self.m.reset_state()



def save_model(model, model_dir):
    """Save model to a binary file.

    This function must write the model to disk in a format that can
    be loaded from the model_fn.

    WARNING - modifying this function may cause the submission process to fail.
    """
    sm_model_dir = os.path.join(getenv('SM_MODEL_DIR'), model_version)
    logger.info(f" model dir is {model_dir}")
    model.save(sm_model_dir)
    
    modelPath = os.path.join(sm_model_dir, 'output')
    if (not os.path.isdir(modelPath)):
        os.makedirs(modelPath)
    if (not os.path.isdir(getenv('SM_MODEL_DIR') + '/code')):
        os.makedirs(getenv('SM_MODEL_DIR') + '/code')

    # Move inference.py so it gets picked up in the archive
    copyfile(os.path.dirname(os.path.realpath(__file__)) + '/inference.py', getenv('SM_MODEL_DIR') + '/code/inference.py')
    copyfile(os.path.dirname(os.path.realpath(__file__)) + '/inference-requirements.txt', getenv('SM_MODEL_DIR') + '/code/requirements.txt')

    with tarfile.open(os.path.join(modelPath, 'model.tar.gz'), mode='x:gz') as archive:
        archive.add(sm_model_dir, recursive=True)

def model_fn(model_dir):
    """Load model from binary file.

    This function loads the model from disk. It is called by SageMaker.

    WARNING - modifying this function may case the submission process to fail.
    """
    model_filepath = os.path.join(model_dir,  model_version)
    logger.info("loading model from " + model_filepath)
    model = keras.models.load_model(model_filepath, custom_objects={"WeightedAccuracy": WeightedAccuracy})
    return model

if __name__ == "__main__":
    """Training Main

    The main function is called by both Unearthed's SageMaker pipeline and the
    Unearthed CLI's "unearthed train" command.
    
    WARNING - modifying this function may cause the submission process to fail.

    The main function must call preprocess, arrange th
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_dir", type=str, default=getenv("SM_MODEL_DIR", "/opt/ml/models")
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=getenv("SM_CHANNEL_TRAINING", "/opt/ml/input/data/training"),
    )
    train(parser.parse_args())
