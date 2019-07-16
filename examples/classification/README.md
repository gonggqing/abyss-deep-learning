# Image Classification

Examples in this directory are to do with training image classification networks based on COCO datasets.


## Training Image Classification Models

This guide covers the following topics:

- Data Loading: How to load in datasets in order to train the models.
- Model Initialisation: How to initialise models.
- Training: How to train the models.
- Data Organisation
- Additional Information: Extras that can help the training procedure.

### Data Loading

For dataset definition, Abyss uses the coco annotation format. The data loaders in this section are designed to take in these coco datasets and load them into the model.
Translating Annotations
Image Level Annotations in COCO datasets

For image classification datasets, image-level annotations are stored as:

- Annotations with a category ID. Having multiple annotations associated with any single image indicates a multiclass dataset. COCO does not provide a use-case for whole image labels, so Abyss labels are stored as a bounding-box or polygon, with zero area (i.e., a (0,0,0,0) bounding box).
- (deprecated) Annotations as a caption. In this case, the label of each annotation is stored as comma separated text in a caption annotation.

#### Category/Caption Maps

Category maps translate the annotations to the output logit of the network. For example, if the category list is as follows:
```python
{
  "categories":
  [
    {
      "id": 1,
      "name": "dog"
    },
    {
      "id": 2,
      "name": "cat"
    }
  ]
}
```



Then an appropriate category map would map the category id, to the output neuron, for example:
```python
{
  "1": 0,  # Maps the category_id for dog to logit 0
  "2": 1   # Maps the category_id for cat to logit 1
}
```

 Category/Caption Maps to alter annotations

Labels often form a hierarchy, for example animal→dog → lakeland terrier. When labelling it is common to label all the way down the hierarchy tree, but then train on an intermediate layer. For example, labelling the breed of dog, but only training to identify dogs.
```python
{
  "categories":
  [
    {
      "id": 1,
      "name": "lakeland_terrier",
      "supercategory": "dog"
    },
    {
      "id": 2,
      "name": "german_shepherd",
      "supercategory": "dog"
    },
    {
      "id": 3,
      "name": "labrador",
      "supercategory": "dog"
    },
    {
      "id": 4,
      "name": "persian",
      "supercategory": "cat"
    },
    {
      "id": 5,
      "name": "siamese",
      "supercategory": "cat"
    },
    {
      "id": 6,
      "name": "sphynx",
      "supercategory": "cat"
    },
  ]
}
```



To train a two class dog vs cat classifier, the following category map would be used. Note the number of classes is given by the number of unique values there are in the category map dictionary.
```python
{
  "1": 0,  # Maps the lakeland_terrier to logit 0 = dog
  "2": 0,  # Maps the german_shepherd to logit 0 = dog
  "3": 0,  # Maps the labrador to logit 0 = dog
  "4": 1,  # Maps the persian to logit 1 = cat
  "5": 1,  # Maps the siamese to logit 1 = cat
  "6": 1   # Maps the sphynx to logit 1 = cat
}
```

#### Annotation Translators

The annotations need to be translated to a vector which is the target of the network. The translation is achieved with an annotation translator. The translator inputs the annotation and transforms this to a list of logit ids, which is then translated to a multihot vector in a later process. These translators can be custom made as long as they subclass AnnotationTranslator. The CategoryTranslator is commonly used when using category format data.

To use the translator, initialise it and then pass it to the ImageClassificationDataset (in the next step).
```python
from abyss_deep_learning.datasets.translators import CategoryTranslator
cat_translator = CategoryTranslator(mapping=cat_map)
```

#### Initialising the dataset

To load this dataset, use the ImageClassificationDataset class. This class inputs a coco dataset and an annotation translator, and provides a generator, which is used to load data on the fly, making it useful for large dataset. The associated data loading of this class happens in the inherited the ClassificationTask, which is responsible for loading the images and the targets.

To initialise the ImageClassificationDataset:
```python
from abyss_deep_learning.datasets.coco import ImageClassificationDataset
train_dataset = ImageClassificationDataset(args.coco_path, translator=cat_translator)
```


To test the data loading, use the sample function:
```python
img, tgt = dataset.sample()

```

To create organize the data into batches, perform augmentation and translate labels to multihot vectors, wrap the generators in a pipeline, which is explained in the next step.


#### Loading Data from the Generators

The train dataset loads the data, but this needs to be operated on in order to make it usable by the models. These operations can include data augmentation, resizing, batching, conversion to multihot.

The pipeline commonly used is:

```python
def pipeline(gen, num_classes, batch_size):
    """
    A sequence of generators that perform operations on the data
    Args:
       gen: the base generator (e.g. from dataset.generator())
       num_classes: (int) the number of classes, to create a multihot vector
       batch_size: (int) the batch size, for the batching generator
   Returns:
   """
   return (batching_gen(lambda_gen(multihot_gen(lambda_gen(gen, func=preprocess), num_classes=num_classes), func=enforce_one_vs_all), batch_size=batch_size))
```

To initialise the pipeline
```python
generator = pipeline(train_dataset.generator(), num_classes=num_classes, batch_size=batch_size)
```

#### Lambda Generators

Lambda Generator is useful for defining custom operations on the data. The user defines a function and passes its handle to lambda generator. The inputs to this function are the image and its caption. For example the preprocess function resizes the image and converts it to float, while passing through the caption.
```python
def preprocess(image, caption):
    """
    A preprocessing function to resize the image
    Args:
       image: (np.ndarray) The image
       caption: passedthrough
    Returns:
       image, caption
    """
    image = resize(image, image_shape, preserve_range=True)
    return preprocess_input(image.astype(NN_DTYPE)), caption
 
# To use the preprocess function as part of a generator
generator = lamda_gen(dataset.generator(), func=preprocess)
```

#### Batching Generators


Models need to operate on batches of data. The dataset doesn't output batches and has doesn't know what the batch size is. The batching generator, accumulates data until it has reach the specified batch size before returning it.
Multihot Generators

The multihot generator converts the output of the annotation translator (which is a list of logit ids) into a multihot vector.

### Model Initialisation

For image classification an ImageClassifier class is used, which initialises, compiles, saves and loads the image classifier model. It uses one of the available keras.applications models, such as Xception.


To initialise an xception model with Imagenet weights
```python
classifier = ImageClassifier(
                 backbone='xception',
                 output_activation='softmax',
                 pooling='avg',
                 classes=num_classes,
                 input_shape=tuple(image_shape),
                 init_weights='imagenet',
                 init_epoch=0,
                 init_lr=args.lr,
                 trainable=True,
                 optimizer='adam',
                 loss='categorical_crossentropy',
                 metrics=['accuracy'],
                 gpus=1,
                 l12_reg=(None, 0.01)
            )
```


Alternatively, to load a model from an existing checkpoint:
```python
classifier = ImageClassifier.load('path/to/checkpoint.h5')

```

### Training

To train the classifer use the 'fit_generator' method. This is preferable/necessary for large datasets. Currently a major bottleneck is the validation process, which can be made significantly faster if the validation data is cached rather than kept as a generator. Use the '--cache-val' argument to do this.

An example use of training the classifier with generators is:
```python
classifier.fit_generator(generator=pipeline(train_gen, num_classes=num_classes, batch_size=args.batch_size),
                         steps_per_epoch=train_steps,
                         validation_data=pipeline(val_gen, num_classes=num_classes, batch_size=args.batch_size) if val_gen else None,
                         validation_steps=val_steps,
                         epochs=args.epochs,
                         verbose=1,
                         shuffle=True,
                         callbacks=callbacks,
                         use_multiprocessing=True,
                         workers=args.workers)
```

### Data Organisation

A clear and consistent method of storing and organising models from training is essential. The following data model should be adhered to at all times.
Scratch Directory

During training, models are typically saved at the end of each epoch. This uses a lot of storage space. The scratch directory is a temporary place to store models and other data generated during training and validation. When the model is validated and is deemed valuable it should be moved to the permanent storage location.

This correct area to save model is
```bash
/mnt/pond/scratch/$PROJECT/$EXPERIMENT_TYPE/$EXPERIMENT
```

For example, for the Abyss Extract project, when training a fault-detection model, the correct path could be:
```bash
/mnt/pond/scratch/cctv-ml-experiments/fault-detection/001.pjm-and-swc

```

#### Naming of Experiments

Experiment names should be ordered and informative. When creating a new experiment, the following naming convention should be used:
```bash
$ENUMERATION.$DATASET_INFO.$EXPERIMENT_INFO1.$EXPERIMENT_INFO2
```

Where:

- ENUMERATION: The experiment number for this experiment type. In the directory, the folder names should start with 001,002,003 etc.
- DATASET_INFO: The dataset used to train this model. For example 'pjm-swc'.
- EXPERIMENT_INFO_N: Any other information about the experiment in shorthand. For example, lr5e-4, which would signify a particular learning rate

Structure of Experiment Directory

The structure of the experiment directory for models in the scratch folder should be:
```bash
├── datasets
    ├── coco.train.json
    ├── coco.val.json
    ├── make-datasets.sh
├── logs
    ├── events.123154.records
├── models
    ├── params.json
    ├── model_definition.json
    ├── model_0.h5
    ├── model_1.h5
├── testing
    ├── interflow
        ├── coco.test.json
        ├── additional-results.png
        ├── testing-scripts.py
    └── quu
        ├── coco.test.json
        ├── additional-results.png
        ├── testing-scripts.py
└── readme.txt

```

The information stored in each of these folders is:

    datasets:
        coco.train.json and coco.val.json, which is used directly when training models
        scripts that are used to reproduce the datasets
    logs:
        log files related to training/validation. Could include tensorboard files and/or csvs.
    models
        Stores all the model checkpoints from training
        Also stores the model_definition.json file and the params.json file
    testing:
        Where the results of testing is stored. Subdirectories indicate different testing conditions, such as different datasets.

### Additional Information

#### Callbacks

##### ImprovedTensorBoard

This callback subclasses and modifies the standard keras tensorboard callback to provide additional metrics such as pr-curves and tfpn metrics.

##### Saving Models

A custom save model callback is currently used, which uses the ImageClassifier.save function. This shouldn't be necessary and using the standard Keras ModelCheckpoint should be fine.

##### Dependencies in current training:

#### Initialising the Optimizer

The optimizer can be initialised from the command line using the --optimizer and --optimizer-args arguments. All the Keras Optimizers are supported. The --optimizer-args is used to further customise the optimizer, such as setting momentum, decay etc. The optimizer-args are a dictionary that has to match the arguments of the optimizer. You don't need to add the learning rate argument as that is added from the --lr argument.

For example to initialise the SGD optimizer with momentum of 0.9:
```bash
python3 train_cctv_classifier.py \
    coco.json \
    --optimizer sgd \
    --optimizer-args {'momentum': 0.9}

```

#### Image Augmentation

The imgaug library is a versatile library that is used for image augmentation. The data can be augmented by using an image augmentation generator and giving it an augmentation configuration. A utility function to create an augmentation configuration is given here. It can be customised from the command line by using a dictionary of arguments. For example:
```bash
python3 train_cctv_classifier.py \
    coco.json \
    --augmentation-configuration {"some_of":None, "flip_lr":True, "flip_ud":True, "gblur":None, "avgblur":None,"gnoise":(0,0.05*255),"scale":(0.8, 1.2), "rotate":(-22.5, 22.5), "bright":(0.75,1.25),"colour_shift":(0.9,1.1)}
```

The augmentation generator is then used as part of the pipeline, for example:
```python
def pipeline(gen, num_classes, batch_size, do_data_aug=False):
     return (batching_gen(augmentation_gen(lambda_gen(multihot_gen(lambda_gen(gen, func=preprocess), num_classes=num_classes),func=enforce_one_vs_all),aug_config=augmentation_cfg, enable=do_data_aug), batch_size=batch_size))
```




