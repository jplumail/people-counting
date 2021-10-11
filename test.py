from train import build_ccnn, create_ds, mae_count, mse_count

import sys
import os
import tensorflow as tf
import tensorflow.keras as K
from PIL import Image
import numpy as np
from matplotlib import cm


checkpoint_path = "data/ShanghaiTech/part_B/runs/211005175533/checkpoints"


# Test the model
filenames = tf.data.Dataset.list_files("data/ShanghaiTech/part_B/test_data/tfrecords/*.tfrecords")
test_ds = filenames.interleave(lambda x: tf.data.TFRecordDataset(x))
test_ds = create_ds(test_ds, batch=1, augment=False)

normalization_layer = K.layers.experimental.preprocessing.Normalization()
normalization_layer.adapt(test_ds.map(lambda x, y: x))

lr = 1
model = build_ccnn(lr, normalization_layer)


def display_dm(epoch, logs):
    im, dm_gt = next(iter(test_ds))
    dm_pred = model.predict(im)
 
    with file_writer_dm.as_default():
        if epoch == 0:
            dm_gt = tf.expand_dims(dm_gt, -1)
            tf.summary.image("image 0", im, max_outputs=5, step=epoch)
            tf.summary.image("density gt 0", dm_gt, max_outputs=5, step=epoch)
        tf.summary.image("density pred 0", dm_pred, max_outputs=5, step=epoch)

dm_callback = K.callbacks.LambdaCallback(on_epoch_end=display_dm)


if os.path.exists(checkpoint_path):
    print("Loading model")
    model = K.models.load_model(checkpoint_path, compile=True, custom_objects={"mae_count": mae_count, "mse_count": mse_count})
    model.compile(optimizer=K.optimizers.Adam(learning_rate=lr), loss="MSE", metrics=[mae_count, mse_count])
else:
    print("Checkpoint not found")
    sys.exit()

mean, var = normalization_layer.mean.numpy(), normalization_layer.variance.numpy()

def show_pred(dm_tensor, img_tensor):
    dm = (dm_tensor.numpy()).astype(float)
    dm = np.clip(dm/0.05, 0, 1)
    img = img_tensor.numpy()

    colormap = cm.get_cmap("plasma")
    dm_color = colormap(dm[...,0])
    dm_rgb = dm_color[...,:3]

    img_to_show = dm_rgb * dm + img * (1 - dm)
    img = Image.fromarray((img_to_show*255).astype(np.uint8), mode="RGB")

    return img

root = "data/ShanghaiTech/part_B/runs/211005175533/test"
os.makedirs(root, exist_ok=True)
for i, (img, dm_gt) in enumerate(test_ds):
    dm_pred = model.predict(img)
    count_pred = dm_pred.sum()
    count_gt = dm_gt.numpy().sum()
    dm_gt = tf.expand_dims(dm_gt, -1)[0]
    dm_pred = dm_pred[0]
    stacked_tensors = tf.stack([dm_gt, dm_pred], axis=0)
    resized_tensors = tf.image.resize(stacked_tensors, (768, 1024), method='bicubic', antialias=True)
    
    img_gt = show_pred(resized_tensors[0], img[0])
    img_pred = show_pred(resized_tensors[1], img[0])
    img = Image.fromarray((img[0].numpy() * 255).astype(np.uint8))

    path_gt = root + "/{}-gt-{:.1f}.png".format(str(i),float(count_gt))
    path_pred = root + "/{}-pred-{:.1f}.png".format(str(i), float(count_pred))
    path = root + "/{}.png".format(str(i))

    img_pred.save(path_pred)
    img_gt.save(path_gt)
    img.save(path)

#model.save(root+"models/ccnn-shanghaitechB")