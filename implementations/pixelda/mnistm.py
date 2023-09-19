"""Dataset setting and data loader for MNIST-M.

Modified from
https://github.com/pytorch/vision/blob/master/torchvision/datasets/mnist.py

CREDIT: https://github.com/corenel
"""

from __future__ import print_function

import errno
import os
import gzip
import pickle

import mindspore
from PIL import Image
from mindspore import Tensor
import mindspore.common.dtype as mstype

# import essential packages
from six.moves import urllib


class MNISTM:
    """`MNIST-M Dataset."""

    url = "https://github.com/VanushVaswani/keras_mnistm/releases/download/1.0/keras_mnistm.pkl.gz"

    raw_folder = "raw"
    processed_folder = "processed"
    training_file = "mnist_m_train.pt"
    test_file = "mnist_m_test.pt"

    mnist_train_labels = []
    mnist_test_labels = []

    def __init__(self, root, mnist_root="data", train=True, transform=None, target_transform=None):
        """Init MNIST-M dataset."""
        super().__init__(MNISTM)
        self.root = os.path.expanduser(root)
        self.mnist_root = os.path.expanduser(mnist_root)
        self.transform = transform
        self.target_transform = target_transform
        self.train = train  # training set or test set

        self.train_ds, self.test_ds = self.dl_preprocess()

    def __getitem__(self, index):
        """Get images and target for data loader.

        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        if self.train:
            img, target = self.train_ds[0][index], self.train_ds[1][index]
        else:
            img, target = self.test_ds[0][index], self.test_ds[1][index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img.squeeze(), mode="RGB")

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img[0], target

    def __len__(self):
        """Return size of dataset."""
        if self.train:
            return len(self.train_ds[0])
        return len(self.test_ds[0])

    def get_train_labels(self, labels):
        """Get MNIST labels"""
        for _, label in enumerate(labels.create_tuple_iterator()):
            self.mnist_train_labels.append(label)

    def get_test_labels(self, labels):
        """Get MNIST labels"""
        for _, label in enumerate(labels.create_tuple_iterator()):
            self.mnist_test_labels.append(label)

    def dl_preprocess(self):
        """Create MNIST-M dataset."""

        # make data dirs
        try:
            os.makedirs(os.path.join(self.root, self.raw_folder))
        except OSError as e:
            if e.errno == errno.EEXIST:
                pass
            else:
                raise

        # download pkl files
        filename = self.url.rpartition("/")[2]
        file_path = os.path.join(self.root, self.raw_folder, filename)
        if not os.path.exists(os.path.join(self.root, 'raw/keras_mnistm.pkl')):
            print("Downloading " + self.url)
            if not os.path.exists(file_path.replace(".gz", "")):
                data = urllib.request.urlopen(self.url)
                with open(file_path, "wb") as f:
                    f.write(data.read())
                with open(file_path.replace(".gz", ""), "wb") as out_f, gzip.GzipFile(file_path) as zip_f:
                    out_f.write(zip_f.read())
                os.unlink(file_path)

        # process and save as torch files
        print("Processing...")

        # load MNIST-M images from pkl file
        with open(file_path.replace(".gz", ""), "rb") as f:
            mnist_m_data = pickle.load(f, encoding="bytes")

        mnist_m_train_data = Tensor(mnist_m_data[b"train"], dtype=mstype.byte)
        mnist_m_test_data = Tensor(mnist_m_data[b"test"], dtype=mstype.byte)

        # get MNIST labels
        mnist_train = mindspore.dataset.MnistDataset(
            dataset_dir=self.mnist_root,
            usage='train',
            shuffle=False)
        mnist_test = mindspore.dataset.MnistDataset(
            dataset_dir=self.mnist_root,
            usage='test',
            shuffle=False)

        for _, (_, label) in enumerate(mnist_train.create_tuple_iterator()):
            self.mnist_train_labels.append(label)

        for _, (_, label) in enumerate(mnist_test.create_tuple_iterator()):
            self.mnist_test_labels.append(label)

        # save MNIST-M dataset
        training_set = (mnist_m_train_data.asnumpy(), self.mnist_train_labels)
        test_set = (mnist_m_test_data.asnumpy(), self.mnist_test_labels)

        print("Done!")
        return training_set, test_set
