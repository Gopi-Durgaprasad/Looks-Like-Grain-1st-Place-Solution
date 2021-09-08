# 1st place solution in Looks Like Grain

Competition Page : https://unearthed.solutions/u/competitions/109/ 

### The Problem
The current visual inspection of grain is performed manually by a person. At the scale CBH operates, this manual inspection can be a time-consuming, resource-intensive process. Through the use of computer vision machine learning, CBH is seeking to increase the efficiency of this inspection process.

### The Challenge
In this challenge, task is correctly classifying images of oat, wheat, barley, and weed grains. Each grain has a range of potential defects that may or may not be present within each image.

Examples:



### MODEL SUMMARY

- The training models what we have used are Efficientnet pretrained models.
- Our best private leaderbord model is an ensemble of three Efficientnet models those are **EfficientNetB0**, **EfficientNetB1** and **EfficientNetB2**.
- It will take 2 hours to train the model in the AWS cluster
- The most important things which have worked for us are augmentations, scheduler, postprocessing and datageneration.

### Team Background

- **Competition Name:** Looks Like Grain
- **Team Name:** Bestfit
- **Private Leaderbord Score:** 0.98727
- **Public Leaderbord Score:** 0.98683


- Name: Gopi Durgaprasad
- Location: India, Andhra Pradesh, Annavaram.
- Email: gopidurgaprasad762@gmail.com


- Name: Shravan Kumar Koninti
- Location: India, Telangana, Hyderabad.
- Email: shravankumar224@gmail.com


### Preprocessing Step
- we are resizing the original images into 224x224x3 
- for resizing we have used PLI and tensorflow image librarys and converted into `uint8` to reduce space.
```python
        img = Image.open(fn)
        img = img.convert('RGB')
        img = img.resize((IMG_SIZE, IMG_SIZE), Image.NEAREST)
        img_array = image.img_to_array(img)
        img_array = img_array.astype(np.uint8)
```
- because of small dataset we loaded training and validation dataset into RAM before training it makes our training faster.

### Data Generation
- For data generation we are using tensorflow custom data generater from `tf.keras.utils.Sequence`.
- it makes our data generation faster and we can apply augmentations effectively. 
- it takes `X`, `y`, `BATCH_SIZE` etc and generates batches of data.

### Augmentations
- We used `Albumentation` library for augmentations.
- we tried lots of complex augmentations but model did NOT converge faster.
- because of less time we needed to converge faster Hence we used simple augmentations.
```python
A.Compose([
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
```
- at the end those simple augmentations work well for us to converge faster.

### Schedulers
- We tried lots of schedulers like `ExponentialDecay`, `CosineDecayRestarts`, `CyclicLR`.
- but at the end `ReduceLROnPlateau` works well for us.
```python
lr_reducer1 = tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", patience=2, min_lr=1e-6, factor=0.1)
```
- if our validation loss did not decrease for 2 epochs we multipy lr with factor `0.1`
- our inital learning rate is `1e-3`.
- those help models to converge faster with in 10 to 15 epochs.

### Models
- we used `Efficientnet` pretraied models.
```python
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
```
- with initial learning rate `1e-3` with `Adam` optimizer
- in efficientnet we used simple models like `E0`, `E1` and `E3`.
- in efficientnet we have larger models also, but because of small data and small imagesize simple models worked well.
- we tried `Densenet`, `ResNet`, `MobileNet`, `Xception` but those are not upto the mark compare to `Efficientnet` models.

### Loss Function
- we used simple and standard `categorical_crossentropy` loss.
- it works very well compared to `focal loss`.
- we also tried weighted `categorical_crossentropy` it helps some extent but it will not work in postprocessing.

### Saving Checkpoints
- we tracked and saved best `val_accuracy` models those gives best results in public leaderbord.
```python
checkpoint = tf.keras.callbacks.ModelCheckpoint(
    save_path, 
    monitor='val_accuracy', 
    verbose=1, 
    save_best_only=True, 
    mode='max', 
    save_weights_only=True
)
```
- we also tracked competition metric and weighted metric `WeightedAccuracy`.

### Ensemble of Models
- we ensemble three simple Efficientnet models those are `EfficientNetB0`, `EfficientNetB1` and `EfficientNetB2` with same training setup.
- sum of those three ensemble models works well.
- we also tried different weighted sum but at the end simple sum works well in private leaderbord.

### Post-processing
- because of competition metric is weighted accuracy no loss function maximize `Accuracy` so post processing works very well for us.
- because we have lots of unbalanced classes and competition metric is self weighted accuracy we multiplied weighted classes with some optimized weights.
- in our metric class `7`, `13` and `24` have weights of `8.94`, so we multiply those classes with optimized weights.
- we find weights that maximize competition metric on validation dataset.
- we are only finding weights for `[7, 13, 24]` becase of those clasess have higher number of images.

### Class Weights
- initially class weights work very well for us.
- we reached `0.9841` without any postprocessing using class weights only.

```python
def w(x):
    if "_SOUND" in x:
        return 8.94 # weighting  Healthy grains more important than defective grains
    return 1

CLASS_WEIGHT = {}
for i, value in enumerate(LBL.keys()):
    CLASS_WEIGHT[i] = w(value)

model.fit( ...., class_weight=CLASS_WEIGHT)
```
- after we comeup with strong postprocessing class weights accuracy less compare to postprocessing.

### Experiments those did not work for us
- high/more/complex augmentations are not work for us.
- cutmix and mixup also not work for us, because of lots of imbalanced classes.
- cyclicLR not work well for faster converge.
- focal loss not working well for us.


### How to improve our solution
- we observed that each class images have very different sizes, some of them even height and width greater then 1000pixels
- so scale those sizes and resize those into high image sizes like 300x300, 500x500 , 600x600.
- try bigger models like `EfficientNetB4`, `EfficientNetB5`, `EfficientNetB6` with different image sizes.
- try to add more augmentations and train for long time those will converge some point.
- without post-processing use `Class Weights`.



