# Taken from https://www.kaggle.com/tthien/shanghaitech-with-people-density-map

import numpy as np
import os
from tensorflow.keras.preprocessing import image
import numpy as np
import scipy.io
import scipy.ndimage
import glob
import time
from joblib import Parallel, delayed
import tensorflow as tf


__DATASET_ROOT = "data/ShanghaiTech/"

def gaussian_filter_density(gt):
    density = np.zeros(gt.shape, dtype=np.float32)
    gt_count = np.count_nonzero(gt)
    if gt_count == 0:
        return density

    pts = np.array(list(zip(np.nonzero(gt)[1], np.nonzero(gt)[0])))

    for i, pt in enumerate(pts):
        pt2d = np.zeros(gt.shape, dtype=np.float32)
        pt2d[pt[1], pt[0]] = 1.
        sigma = 15.0 # Constant sigma
        density += scipy.ndimage.filters.gaussian_filter(pt2d, sigma, mode='constant')
    return density


def generate_density_map(img_path):
    mat_path = img_path.replace('.jpg', '.mat').replace('images', 'ground-truth').replace('IMG_', 'GT_IMG_')
    mat = scipy.io.loadmat(mat_path)
    imgfile = image.load_img(img_path)
    img = image.img_to_array(imgfile)
    k = np.zeros((img.shape[0], img.shape[1]))
    gt = mat["image_info"][0, 0][0, 0][0]
    for i in range(0, len(gt)):
        if int(gt[i][1]) < img.shape[0] and int(gt[i][0]) < img.shape[1]:
            k[int(gt[i][1]), int(gt[i][0])] = 1
    k = gaussian_filter_density(k)

    # To tensors
    img = img / 255. # Normalization
    concat_array = np.concatenate([img, k[:,:,None]], axis=2)
    concat_tensor = tf.convert_to_tensor(concat_array, dtype=tf.float32)
    x = tf.io.serialize_tensor(concat_tensor)

    feature = {"data": tf.train.Feature(
        bytes_list=tf.train.BytesList(value=[x.numpy()])
    )}
    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))

    return example_proto.SerializeToString()


def feature_generator(img_path_list):
    num_sample_per_file = 10
    n_jobs = 3
    
    file_path = os.path.join(os.path.dirname(img_path_list[0]).replace("images", "tfrecords"), "{}.tfrecords")
    file_idx = 0
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    paths = iter(img_path_list)
    
    stop = False
    while stop is False:
        tfrecord_writer = tf.io.TFRecordWriter(file_path.format(file_idx))
        try:
            paths_to_process = [next(paths) for i in range(num_sample_per_file)]
        except StopIteration:
            stop = True
        serial_samples = Parallel(n_jobs=n_jobs)(delayed(generate_density_map)(path) for path in paths_to_process)
        for sample in serial_samples:
            tfrecord_writer.write(sample)
        file_idx += 1

def generate_shanghaitech_path(root):
    part_A_train = os.path.join(root, 'part_A', 'train_data', 'images')
    part_A_test = os.path.join(root, 'part_A', 'test_data', 'images')
    part_B_train = os.path.join(root, 'part_B', 'train_data', 'images')
    part_B_test = os.path.join(root, 'part_B', 'test_data', 'images')
    path_sets = [part_A_train, part_A_test, part_B_train, part_B_test]

    list_paths = [[] for _ in path_sets]
    for i, path in enumerate(path_sets):
        for img_path in glob.glob(os.path.join(path, '*.jpg')):
            print(img_path)
            list_paths[i].append(img_path)

    return list_paths


if __name__ == "__main__":

    start_time = time.time()
    a_train, a_test, b_train, b_test = generate_shanghaitech_path(__DATASET_ROOT)

    # Generate only for the Shanghai part B
    feature_generator(b_train)
    feature_generator(b_test)

    print("--- %s seconds ---" % (time.time() - start_time))