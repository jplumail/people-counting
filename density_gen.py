# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in

import numpy as np  # linear algebra

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os

# Any results you write to the current directory are saved as output.


__author__ = "Thai Thien"
__email__ = "tthien@apcs.vn"

import os
from tensorflow.keras.preprocessing import image
import numpy as np
import scipy
from scipy.io import loadmat
import glob
import h5py
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
    #leafsize = 2048
    # build kdtree
    #pts_copy = pts.copy()
    #tree = scipy.spatial.KDTree(pts_copy, leafsize=leafsize)
    # query kdtree
    #distances, locations = tree.query(pts, k=4)

    for i, pt in enumerate(pts):
        pt2d = np.zeros(gt.shape, dtype=np.float32)
        pt2d[pt[1], pt[0]] = 1.
        #if gt_count > 1:
        #    sigma = (distances[i][1] + distances[i][2] + distances[i][3]) * 0.1
        #else:
        #    sigma = np.average(np.array(gt.shape)) / 2. / 2.  # case: 1 point
        sigma = 15.0
        density += scipy.ndimage.filters.gaussian_filter(pt2d, sigma, mode='constant')
    return density


def single_sample_prototype():
    img_path = '/data/dump/ShanghaiTech/part_A/train_data/images/IMG_2.jpg'
    mat_path = "/data/dump/ShanghaiTech/part_A/train_data/ground-truth/GT_IMG_2.mat"
    mat = scipy.io.loadmat(mat_path)
    imgfile = image.load_img(img_path)
    img = image.img_to_array(imgfile)
    k = np.zeros((img.shape[0], img.shape[1]))
    gt = mat["image_info"][0, 0][0, 0][0]
    for i in range(0, len(gt)):
        if int(gt[i][1]) < img.shape[0] and int(gt[i][0]) < img.shape[1]:
            k[int(gt[i][1]), int(gt[i][0])] = 1
    k = gaussian_filter_density(k)


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

    file_path = "/".join(img_path_list[0].split("/")[:-1]).replace("images", "tfrecords") + "/{}.tfrecords"
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
    # now generate the ShanghaiA's ground truth
    part_A_train = os.path.join(root, 'part_A/train_data', 'images')
    part_A_test = os.path.join(root, 'part_A/test_data', 'images')
    part_B_train = os.path.join(root, 'part_B/train_data', 'images')
    part_B_test = os.path.join(root, 'part_B/test_data', 'images')
    path_sets = [part_A_train, part_A_test, part_B_train, part_B_test]

    img_paths_a_train = []
    img_paths_a_test = []
    img_paths_b_train = []
    img_paths_b_test = []

    for img_path in glob.glob(os.path.join(part_A_train, '*.jpg')):
        img_paths_a_train.append(img_path)
    for img_path in glob.glob(os.path.join(part_B_train, '*.jpg')):
        img_paths_b_train.append(img_path)

    for img_path in glob.glob(os.path.join(part_A_test, '*.jpg')):
        img_paths_a_test.append(img_path)
    for img_path in glob.glob(os.path.join(part_B_test, '*.jpg')):
        img_paths_b_test.append(img_path)

    return img_paths_a_train, img_paths_a_test, img_paths_b_train, img_paths_b_test


if __name__ == "__main__":
    """
    TODO: this file will preprocess crowd counting dataset
    """

    start_time = time.time()
    a_train, a_test, b_train, b_test = generate_shanghaitech_path(__DATASET_ROOT)

    feature_generator(b_train)
    feature_generator(b_test)

    print("--- %s seconds ---" % (time.time() - start_time))