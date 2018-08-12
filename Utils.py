from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import random
import os
import sys
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.contrib.learn.python.learn.datasets import base
from tensorflow.python.framework import random_seed
import glob2

distributions = tf.contrib.distributions

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.01)

  return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.0, shape=shape)

    return tf.Variable(initial)

#We map past locations (mean_arr) to some distribution over potential locations at the current time step
#and return the log likelihood of these possible samples
def loglikelihood(mean_arr, sampled_arr, sigma):
    mu = tf.stack(mean_arr)  # mu = [timesteps, batch_size, loc_dim]
    sampled = tf.stack(sampled_arr)  # same shape as mu
    gaussian = distributions.Normal(mu, sigma)
    logll = gaussian.log_prob(sampled)  # [timesteps, batch_size, loc_dim]
    logll = tf.reduce_sum(logll, 2)
    logll = tf.transpose(logll)  # [batch_size, timesteps]

    return logll

class DataSet(object):
    def __init__(self,
               images,
               labels,
               reshape=True,
               seed=None):
        seed1, seed2 = random_seed.get_seed(seed)
        # If op level seed is not set, use whatever graph level seed is returned
        np.random.seed(seed1 if seed is None else seed2)
        assert images.shape[0] == labels.shape[0], (
            'images.shape: %s labels.shape: %s' % (images.shape, labels.shape))
        self._num_examples = images.shape[0]

        # Convert shape from [num examples, rows, columns, depth]
        # to [num examples, rows*columns]
        if reshape:
            images = images.reshape(images.shape[0],
                                    images.shape[1] * images.shape[2])

        self._images = images
        self._labels = labels
        self._epochs_completed = 0
        self._index_in_epoch = 0

    @property
    def images(self):
        return self._images

    @property
    def labels(self):
        return self._labels

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def epochs_completed(self):
        return self._epochs_completed

    def next_batch(self, batch_size, shuffle=True):
        """Return the next `batch_size` examples from this data set."""
        start = self._index_in_epoch
        # Shuffle for the first epoch
        if self._epochs_completed == 0 and start == 0 and shuffle:
            perm0 = np.arange(self._num_examples)
            np.random.shuffle(perm0)
            self._images = self.images[perm0]
            self._labels = self.labels[perm0]
        # Go to the next epoch
        if start + batch_size > self._num_examples:
            # Finished epoch
            self._epochs_completed += 1
            # Get the rest examples in this epoch
            rest_num_examples = self._num_examples - start
            images_rest_part = self._images[start:self._num_examples]
            labels_rest_part = self._labels[start:self._num_examples]
            # Shuffle the data
            if shuffle:
                perm = np.arange(self._num_examples)
                np.random.shuffle(perm)
                self._images = self.images[perm]
                self._labels = self.labels[perm]
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size - rest_num_examples
            end = self._index_in_epoch
            images_new_part = self._images[start:end]
            labels_new_part = self._labels[start:end]
            return np.concatenate(
                (images_rest_part, images_new_part), axis=0), np.concatenate(
                    (labels_rest_part, labels_new_part), axis=0)
        else:
            self._index_in_epoch += batch_size
            end = self._index_in_epoch
            return self._images[start:end], self._labels[start:end]

def extract_pc_and_labels(data_dir, 
                          num_points_per_pc, 
                          all_examples, 
                          all_labels, 
                          examples_indices):
    len_data = len(examples_indices)
    data = np.empty((len_data, num_points_per_pc, 3))
    labels = np.zeros((len_data,))

    for i, index in enumerate(examples_indices):
        curr_example = all_examples[index]

        curr_file = os.path.join(data_dir, curr_example)
        curr_pc = open(curr_file, 'r').read().split('\n')
        rows = [i for i in curr_pc if i != '']
        all_points = [i.split(',')[:3] for i in rows]
        all_points = np.array(all_points, dtype=np.float32)

        data[i] = all_points
        
        curr_label = curr_example.split('/')[0]
        one_hot_pos = np.where(all_labels == curr_label)
        labels[i] = one_hot_pos[0][0]

    return data, labels

def read_data_sets(data_dir,
                   num_points_per_pc,
                   reshape=True,
                   seed=None):
    try:
        print('Trying to load existing ModelNet40 datasets...')
        options = dict(reshape=reshape, seed=seed)

        train_pc = np.load('train_pc.npy')
        train_labels = np.load('train_labels.npy')
        val_pc = np.load('val_pc.npy')
        val_labels = np.load('val_labels.npy')
        test_pc = np.load('test_pc.npy')
        test_labels = np.load('test_labels.npy')

        print('Building train, validation and test dataset objects...')

        train = DataSet(train_pc, train_labels, **options)
        validation = DataSet(val_pc, val_labels, **options)
        test = DataSet(test_pc, test_labels, **options)

        return base.Datasets(train=train, validation=validation, test=test)
    except:
        print('No existing ModelNet40 train, validation and test sets found...Rebuilding')
        
        example_list = open(data_dir + '/filelist.txt', 'r')
        all_examples = example_list.read().split('\n')
        all_examples = [i for i in all_examples if i != '']
        all_examples = np.array(all_examples)
        example_list.close()

        num_examples = len(all_examples)
        range_examples = np.arange(0, num_examples)
        random.Random(0).shuffle(range_examples)
        
        label_list = open(data_dir + '/modelnet40_shape_names.txt', 'r')
        all_labels = label_list.read().split('\n')
        all_labels = [i for i in all_labels if i != '']
        all_labels = np.array(all_labels)
        label_list.close()

        X_train, X_test, _, _ = train_test_split(range_examples, 
                                                range_examples, 
                                                test_size=0.2, 
                                                random_state=42)

        X_val, X_test, _, _ = train_test_split(X_test, 
                                            X_test, 
                                            test_size=0.5, 
                                            random_state=42)

        train_pc, train_labels = extract_pc_and_labels(data_dir, 
                                                        num_points_per_pc, 
                                                        all_examples, 
                                                        all_labels, 
                                                        X_train)
        val_pc, val_labels = extract_pc_and_labels(data_dir,
                                                    num_points_per_pc, 
                                                    all_examples, 
                                                    all_labels, 
                                                    X_val)
        test_pc, test_labels = extract_pc_and_labels(data_dir,
                                                        num_points_per_pc, 
                                                        all_examples, 
                                                        all_labels, 
                                                        X_test)

        options = dict(reshape=reshape, seed=seed)

        np.save('train_pc', train_pc)
        np.save('train_labels', train_labels)
        np.save('val_pc', val_pc)
        np.save('val_labels', val_labels)
        np.save('test_pc', test_pc)
        np.save('test_labels', test_labels)

        print('Building train, validation and test dataset objects...')

        train = DataSet(train_pc, train_labels, **options)
        validation = DataSet(val_pc, val_labels, **options)
        test = DataSet(test_pc, test_labels, **options)

        return base.Datasets(train=train, validation=validation, test=test)
