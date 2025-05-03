from __future__ import division, print_function, absolute_import
import numpy as np
import collections
import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import scipy.io as scio
import h5py


# from matplotlib import pyplot as plt


def dense_to_one_hot(labels_dense, num_classes):
    """Convert class labels from scalars to one-hot vectors."""
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot


def read_dataset(data_dir, name='soc', mode='train', viewnum=2, num_classes=10, dtypes='float32'):
    img = data_dir + str(viewnum) + 'views/' + 'img_' + mode + '_' + str(name) + '_' + str(viewnum) + 'view.mat'
    label = data_dir + str(viewnum) + 'views/' + 'label_' + mode + '_' + str(name) + '_' + str(viewnum) + 'view.mat'
    # train
    # data = scio.loadmat(img_train)
    data = h5py.File(img)
    img = data['img_' + mode]  # size 2-view:21 834, 3-view:48 764, 4-view: 43 533
    # img_train = tf.transpose(img_train, perm = [3, 2, 1, 0])

    # data = scio.loadmat(label_train)
    data = h5py.File(label)
    label = data['label_' + mode]
    # label = tf.transpose(label, perm = [1, 0])
    label = dense_to_one_hot(np.array(label, dtype=np.uint8), num_classes=num_classes)

    dataset = MyDataSet(img, label, dtypes=dtypes)
    return dataset


class MyDataSet(Dataset):
    def __init__(self, images, labels, dtypes='float64'):
        self._images = np.array(images, dtype=dtypes)
        self._labels = np.array(labels)
        self.transform = transforms.Compose([transforms.ToTensor()])

    def __len__(self):
        return len(self._labels)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        _image = self._images[idx]
        _label = self._labels[idx]

        if self.transform:
            _image = self.transform(_image)

        return _image, _label

