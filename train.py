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
    'data': tf.io.FixedLenFeature([], tf.string),
}

def _parse_example_function(example_proto):
  # Parse the input tf.train.Example proto using the dictionary above.
  return tf.io.parse_single_example(example_proto, image_feature_description)

def to_tensor(datapoint):
    data = tf.io.parse_tensor(datapoint['data'], tf.float32)
    data = tf.ensure_shape(data, [None, None, 4])
    return data

def resize_density(data):
    image, density = data[...,:-1], data[...,-1:]
    shape = tf.shape(density)
    newshape = [shape[-3]//8, shape[-2]//8]
    s1 = tf.reduce_sum(density, keepdims=True, axis=(-3,-2))
    density = tf.image.resize(density, newshape, method='bicubic', antialias=True)
    s2 = tf.reduce_sum(density, keepdims=True, axis=(-3,-2))
    density = s1 * tf.math.divide_no_nan(density, s2)
    return image, density[...,0]

HEIGHT = 512
WIDTH = 512

def augment_data(data):
    shape = tf.shape(data)
    if len(shape) == 4:
        batch_size = shape[0]
        batch = True
    else:
        batch_size = 1
        data = tf.expand_dims(data, 0)
        batch = False
    h, w = shape[-3], shape[-2]
    # Random crop
    if h < HEIGHT or w < WIDTH:
        data = tf.image.resize(data, (HEIGHT, WIDTH))
    else:
        data = tf.image.random_crop(data, (batch_size, HEIGHT, WIDTH, 4))
    
    # Random flip left-right
    data = tf.image.random_flip_left_right(data)
    
    if not batch:
        data = data[0]

    return data

def create_tfrecords(root):
    for set_ in ["train", "test"]:
        create_data(os.path.join(root, "{}_data".format(set_)))

def create_ds(ds, cache=False, shuffle=False, batch=1, augment=False):
    ds = ds.map(_parse_example_function, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.map(to_tensor, num_parallel_calls=tf.data.AUTOTUNE)
    if cache:
        ds = ds.cache()
    if shuffle:
        ds = ds.shuffle(100)
    ds = ds.batch(batch)
    if augment: # Change the size of the data
        ds = ds.map(augment_data, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.map(resize_density, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds

def split_ds(ds):
    val_size = int(IMAGE_COUNT * 0.2)
    train_dataset = ds.skip(val_size)
    valid_dataset = ds.take(val_size)
    return train_dataset, valid_dataset


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


if __name__ == "__main__":

    if "DATA_DIR" in os.environ:
        root = os.environ["DATA_DIR"] + "/"
    else:
        root = "data/"

    filenames_train = tf.data.Dataset.list_files(root+"ShanghaiTech/part_B/train_data/tfrecords/*.tfrecords").shuffle(10)
    filenames_valid = tf.data.Dataset.list_files(root+"ShanghaiTech/part_B/test_data/tfrecords/*.tfrecords")
    train_ds = filenames_train.interleave(lambda x: tf.data.TFRecordDataset(x))
    valid_ds = filenames_valid.interleave(lambda x: tf.data.TFRecordDataset(x))


    if "BATCH_SIZE" in os.environ:
        batch_size = int(os.environ["BATCH_SIZE"])
    else:
        batch_size = 8
    
    train_ds = create_ds(train_ds, cache=False, shuffle=True, batch=batch_size, augment=True)
    valid_ds = create_ds(valid_ds, cache=False, batch=8, augment=False).cache()

    normalization_layer = K.layers.experimental.preprocessing.Normalization()
    normalization_layer.adapt(train_ds.map(lambda x, y: x))

    if "CHECKPOINT" in os.environ:
        ts = str(os.environ["CHECKPOINT"])
    else:
        ts = datetime.datetime.now().strftime("%y%m%d%H%M%S")

    run_path = root + "ShanghaiTech/part_B/runs/" + ts + "/"
    log_dir = run_path + "logs/"
    file_writer_dm = tf.summary.create_file_writer(log_dir+"density_map/")
    
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=0, profile_batch=(75,125), write_graph=False)


    checkpoint_path = run_path + "checkpoints/"
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        monitor='val_loss',
        mode='min',
        save_weights_only=False,
        verbose=1
    )
    


    dm_callback = K.callbacks.LambdaCallback(on_epoch_end=display_dm)

    if "LR" in os.environ:
        lr = float(os.environ["LR"])
    else:
        lr = 1e-5

    if "INITIAL_EPOCH" in os.environ:
        initial_epoch = int(os.environ["INITIAL_EPOCH"])
    else:
        initial_epoch = 0

    if "NB_EPOCHS" in os.environ:
        nb_epochs = int(os.environ["NB_EPOCHS"])
    else:
        nb_epochs = 50000

    # model = build_lcnn(lr, normalization_layer)
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
        epochs=nb_epochs
    )