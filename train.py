import tensorflow as tf
import numpy as np
import tensorflow.keras as K
from PIL import Image
import datetime

import os


IMAGE_COUNT = 400


def count(density_map):
    return tf.math.reduce_sum(density_map, axis=(1,2))

def mse_count(density_map_true, density_map_pred):
    return tf.math.reduce_mean(((count(density_map_true) - count(density_map_pred)) ** 2))

def mae_count(density_map_true, density_map_pred):
    return tf.math.reduce_mean(tf.math.abs(count(density_map_true) - count(density_map_pred)))

image_feature_description = {
    'height': tf.io.FixedLenFeature([], tf.int64),
    'width': tf.io.FixedLenFeature([], tf.int64),
    'image': tf.io.FixedLenFeature([], tf.string),
    'points': tf.io.FixedLenFeature([], tf.string),
    'density_map': tf.io.FixedLenFeature([], tf.string),
}

def _parse_image_function(example_proto):
  # Parse the input tf.train.Example proto using the dictionary above.
  return tf.io.parse_single_example(example_proto, image_feature_description)

def to_tensor(datapoint):
    image_raw = datapoint['image']
    image = tf.image.decode_jpeg(image_raw)
    if tf.shape(image)[2] == 1:
        image = tf.image.grayscale_to_rgb(image)

    image = tf.cast(image, tf.float32) / 255.0 # Normalization
    image = tf.ensure_shape(image, [None, None, 3])

    density = tf.io.parse_tensor(datapoint['density_map'], tf.float64)
    density = tf.ensure_shape(density, [None, None])
    density = tf.cast(density, tf.float32)
    
    return image, density

def resize_density(image, density):
    shape = tf.shape(density)
    newshape = [shape[-2]//8, shape[-1]//8]
    density = tf.expand_dims(density, axis=-1)
    s1 = tf.reduce_sum(density, keepdims=True, axis=(-3,-2))
    density = tf.image.resize(density, newshape, method='bicubic', antialias=True)
    s2 = tf.reduce_sum(density, keepdims=True, axis=(-3,-2))
    density = s1 * tf.math.divide_no_nan(density, s2)
    return image, density[...,0]

HEIGHT = 512
WIDTH = 512

def augment_data(image, density):
    density = tf.expand_dims(density, axis=-1) # add a the channel dim
    inputs = tf.concat([image, density], axis=-1)
    
    shape = tf.shape(image)
    if len(shape) == 4:
        batch_size = shape[0]
        batch = True
    else:
        batch_size = 1
        inputs = tf.expand_dims(inputs, 0)
        batch = False
    h, w = shape[-3], shape[-2]
    # Random crop
    if h < HEIGHT or w < WIDTH:
        inputs = tf.image.resize(inputs, (HEIGHT, WIDTH))
    else:
        inputs = tf.image.random_crop(inputs, (batch_size, HEIGHT, WIDTH, 4))
    
    # Random flip left-right
    inputs = tf.image.random_flip_left_right(inputs)
    
    out_im = inputs[...,:3]
    out_density = inputs[...,3]

    if not batch:
        out_im = out_im[0]
        out_density = out_density[0]

    return out_im, out_density

def create_tfrecords(root):
    for set_ in ["train", "test"]:
        create_data(os.path.join(root, "{}_data".format(set_)))

def create_ds(ds, cache=False, shuffle=False, batch=1, augment=False):
    ds = ds.map(_parse_image_function, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.map(to_tensor, num_parallel_calls=tf.data.AUTOTUNE)
    if cache:
        ds = ds.cache()
    if shuffle:
        ds = ds.shuffle(100)
    ds = ds.batch(batch)
    if augment:
        ds = ds.map(augment_data, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.map(resize_density, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds

def split_ds(ds):
    val_size = int(IMAGE_COUNT * 0.2)
    train_dataset = ds.skip(val_size)
    valid_dataset = ds.take(val_size)
    return train_dataset, valid_dataset

root = os.environ["DATA_DIR"]


# Commented out IPython magic to ensure Python compatibility.
# %load_ext tensorboard
# %tensorboard --logdir logs/fit

filenames_train = tf.data.Dataset.list_files(root+"ShanghaiTechB/train_data/data-train*.tfrecords").shuffle(10)
filenames_valid = tf.data.Dataset.list_files(root+"ShanghaiTechB/train_data/data-valid*.tfrecords")
train_ds = filenames_train.interleave(lambda x: tf.data.TFRecordDataset(x))
valid_ds = filenames_valid.interleave(lambda x: tf.data.TFRecordDataset(x))
 
train_ds = create_ds(train_ds, cache=True, shuffle=True, batch=32, augment=True)
valid_ds = create_ds(valid_ds, cache=True, batch=32, augment=False).cache()

normalization_layer = K.layers.experimental.preprocessing.Normalization()
normalization_layer.adapt(train_ds.map(lambda x, y: x))

if "CHECKPOINT" in os.environ:
    ts = str(os.environ["CHECKPOINT"])
else:
    ts = datetime.datetime.now().strftime("%s")

log_dir = root + "ShanghaiTechB/logs/" + ts + "/"
file_writer_dm = tf.summary.create_file_writer(log_dir+"/density_map/")
 
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=0, profile_batch=0, write_graph=False)
 
checkpoint_path = root + "ShanghaiTechB/models/" + ts + "/"
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path,
    monitor='val_loss',
    mode='min',
    save_weights_only=False,
    verbose=1
)
 

# https://arxiv.org/abs/1907.02198
def build_lcnn(lr, preprocessing_layer):
    lcnn = K.models.Sequential(name="LCNN")
    lcnn.add(preprocessing_layer)
    lcnn.add(K.layers.Conv2D(8, 3, padding='same'))
    lcnn.add(K.layers.Conv2D(16, 3, padding='same'))
    lcnn.add(K.layers.MaxPool2D(pool_size=(2,2)))
    lcnn.add(K.layers.Conv2D(32, 3, padding='same'))
    lcnn.add(K.layers.Conv2D(32, 3, padding='same'))
    lcnn.add(K.layers.Conv2D(32, 3, padding='same'))
    lcnn.add(K.layers.Conv2D(32, 3, padding='same'))
    lcnn.add(K.layers.MaxPool2D(pool_size=(2,2)))
    lcnn.add(K.layers.Conv2D(64, 3, padding='same'))
    lcnn.add(K.layers.Conv2D(64, 3, padding='same'))
    lcnn.add(K.layers.Conv2D(64, 3, padding='same'))
    lcnn.add(K.layers.Conv2D(64, 3, padding='same'))
    lcnn.add(K.layers.Conv2D(64, 3, padding='same'))
    lcnn.add(K.layers.Conv2D(64, 3, padding='same'))
    lcnn.add(K.layers.MaxPool2D(pool_size=(2,2)))
    lcnn.add(K.layers.Conv2D(32, 3, padding='same'))
    lcnn.add(K.layers.Conv2D(32, 3, padding='same'))
    lcnn.add(K.layers.Conv2D(16, 3, padding='same'))
    lcnn.add(K.layers.Conv2D(16, 3, padding='same'))
    lcnn.add(K.layers.Conv2D(8, 3, padding='same'))
    lcnn.add(K.layers.Conv2D(1, 1, padding='same'))
    #lcnn.add(SumLayer(axis=(1,2,3)))
    lcnn.build((None, None, None, 3))
 
    lcnn.compile(optimizer=K.optimizers.Adam(learning_rate=lr), loss="MSE", metrics=[mae_count, mse_count])
    return lcnn
 
# https://arxiv.org/abs/2002.06515
def build_ccnn(lr, preprocessing_layer):
    im = K.layers.Input(shape=(None, None, 3))
    out = preprocessing_layer(im)
    out1 = K.layers.Conv2D(10, 9, padding='same')(out)
    out2 = K.layers.Conv2D(14, 7, padding='same')(out)
    out3 = K.layers.Conv2D(16, 5, padding='same')(out)
    features = K.layers.Concatenate()([out1, out2, out3])
    
    layers = K.models.Sequential()
    layers.add(K.layers.MaxPool2D(pool_size=(2,2)))
    layers.add(K.layers.Conv2D(60, 3, padding='same'))
    layers.add(K.layers.MaxPool2D(pool_size=(2,2)))
    layers.add(K.layers.Conv2D(40, 3, padding='same'))
    layers.add(K.layers.MaxPool2D(pool_size=(2,2)))
    layers.add(K.layers.Conv2D(20, 3, padding='same'))
    layers.add(K.layers.Conv2D(10, 3, padding='same'))
    layers.add(K.layers.Conv2D(1, 1, padding='same'))
 
    out = layers(features)
    ccnn = K.Model(inputs=im, outputs=out, name="C-CNN")
 
    ccnn.compile(optimizer=K.optimizers.Adam(learning_rate=lr), loss="MSE", metrics=[mae_count, mse_count])
    return ccnn
 
 
def display_dm(epoch, logs):
    im, dm_gt = next(iter(valid_ds))
    dm_pred = model.predict(im)
 
    with file_writer_dm.as_default():
        if epoch == 0:
            dm_gt = tf.expand_dims(dm_gt, -1)
            tf.summary.image("image 0", im, max_outputs=5, step=epoch)
            tf.summary.image("density gt 0", dm_gt, max_outputs=5, step=epoch)
        tf.summary.image("density pred 0", dm_pred, max_outputs=5, step=epoch)
 
dm_callback = K.callbacks.LambdaCallback(on_epoch_end=display_dm)

if "LR" in os.environ:
    lr = float(os.environ["LR"])
else:
    lr = 1e-5

if "INITIAL_EPOCH" in os.environ:
    initial_epoch = int(os.environ["INITIAL_EPOCH"])
else:
    initial_epoch = 0

model = build_ccnn(lr, normalization_layer)
 
if os.path.exists(checkpoint_path):
    print("Loading model")
    model = K.models.load_model(checkpoint_path, compile=True, custom_objects={"mae_count": mae_count, "mse_count": mse_count})
    model.compile(optimizer=K.optimizers.Adam(learning_rate=lr), loss="MSE", metrics=[mae_count, mse_count])
 
model.fit(
    train_ds,
    validation_data=valid_ds,
    callbacks=[model_checkpoint_callback, tensorboard_callback, dm_callback],
    initial_epoch=initial_epoch,
    epochs=10000
)


# Test the model
filenames = tf.data.Dataset.list_files(root+"ShanghaiTechB/test_data/*.tfrecords")
test_ds = filenames.interleave(lambda x: tf.data.TFRecordDataset(x))
test_ds = create_ds(test_ds, batch=1, augment=False)

model.evaluate(test_ds)

model.save(root+"models/ccnn-shanghaitechB")
