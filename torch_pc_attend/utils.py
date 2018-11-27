from __future__ import division

import json
import os
import sys
import pickle
import arrow

import numpy as np
import torch
from torch.utils import data

def text_to_torch(path_to_file):
    file = open(path_to_file, 'r')
    all_points = file.read().split('\n')
    points = [i for i in all_points if i != '']

    xs = []
    ys = []
    zs = []
    for point in points:
        points_sep = point.split(',')
        x = float(points_sep[0])
        y = float(points_sep[1])
        z = float(points_sep[2])

        xs.append(x)
        ys.append(y)
        zs.append(z)

    point_cloud = np.vstack((np.array(xs).T, np.array(ys).T, np.array(zs).T))
    return torch.from_numpy(point_cloud)

def write_torch_dataset(modelnet_dir):
    if (os.path.exists('./data')):
        return

    print('Loading ModelNet40 files...')
    os.makedirs('./data')
    # Read in the ModelNet labels
    label_list = open(modelnet_dir + '/modelnet40_shape_names.txt', 'r')
    all_labels = label_list.read().split('\n')
    all_labels = [i for i in all_labels if i != '']
    label_list.close()

    # Read in train and test sets
    train_list = open(modelnet_dir + '/modelnet40_train.txt', 'r')
    train_files = train_list.read().split('\n')
    train_files = [i for i in train_files if i != '']
    train_list.close()

    test_list = open(modelnet_dir + '/modelnet40_test.txt', 'r')
    test_files = test_list.read().split('\n')
    test_files = [i for i in test_files if i != '']
    test_list.close()

    # Write the train and test PyTorch tensors
    print('Writing train and test PyTorch tensors...')
    partition_dict = dict()
    labels_dict = dict()

    i = 0
    total = len(train_files) + len(test_files)
    train_labels = []
    for file in train_files:
        folder = file[:-5]
        path_to_file = os.path.join(modelnet_dir, folder, file + '.txt')
        torch_tensor = text_to_torch(path_to_file)
        torch.save(torch_tensor, './data/' + str(i) + '.pt')

        labels_dict[str(i)] = all_labels.index(folder)
        train_labels.append(i)
        i += 1
        print('Completed: %.2f %%' % (i * 100 / total))

    partition_dict['train'] = train_labels

    test_labels = []
    for file in test_files:
        folder = file[:-5]
        path_to_file = os.path.join(modelnet_dir, folder, file + '.txt')
        torch_tensor = text_to_torch(path_to_file)
        torch.save(torch_tensor, './data/' + str(i) + '.pt')

        labels_dict[str(i)] = all_labels.index(folder)
        test_labels.append(i)
        i += 1
        print('Completed: %.2f %%' % (i * 100 / total))

    partition_dict['test'] = test_labels

    print('Writing dictionaries...')
    with open('./partition_dict.txt', 'wb') as p:
        pickle.dump(partition_dict, p, protocol=-1)
    with open('./labels_dict.txt', 'wb') as l:
        pickle.dump(labels_dict, l, protocol=-1)

def prepare_dirs(config):
    for path in [config.data_dir, config.ckpt_dir, config.logs_dir]:
        if not os.path.exists(path):
            os.makedirs(path)

def save_config(config):
    model_name = 'ram_{}_{}_{}_{}'.format(
            config.num_samples, config.num_points_per_sample, config.glimpse_scale, config.init_lr
    )
    filename = model_name + '_params.json'
    param_path = os.path.join(config.ckpt_dir, filename)

    print("[*] Model Checkpoint Dir: {}".format(config.ckpt_dir))
    print("[*] Param Path: {}".format(param_path))

    with open(param_path, 'w') as fp:
        json.dump(config.__dict__, fp, indent=4, sort_keys=True)

def prep(config):
    pass

class AverageMeter(object):
    """
    Computes and stores the average and
    current value.
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class PointCloud(data.Dataset):
    """
    Args:
        root (string): Root directory of dataset where ``processed/training.pt``
            and  ``processed/test.pt`` exist.
        train (bool, optional): If True, creates dataset from ``training.pt``,
            otherwise from ``test.pt``.
    """
    def __init__(self, partition_dict, labels_dict, train=True):
        self.train = train  # training set or test set

        with open(partition_dict, 'rb') as p:
            partition = pickle.load(p)
        with open(labels_dict, 'rb') as l:
            labels = pickle.load(l)

        self.labels = labels
        self.partition = partition

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        if self.train:
            list_IDs = self.partition['train']
        else:
            list_IDs = self.partition['test']

        ID = str(list_IDs[index])
        img = torch.load('./data/' + ID + '.pt')
        target = self.labels[ID]

        return img, target

    def __len__(self):
        if self.train:
            return len(self.partition['train'])
        else:
            return len(self.partition['test'])
