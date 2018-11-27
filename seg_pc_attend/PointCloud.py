from __future__ import division

import os

import torch
from torch.utils import data


class PointCloud(data.Dataset):
    """
    Args:
        root (string): Root directory of dataset where ``processed/training.pt``
            and  ``processed/test.pt`` exist.
        train (bool, optional): If True, creates dataset from ``training.pt``,
            otherwise from ``test.pt``.
    """
    def __init__(self, semantic_dir, semi, pc_size, binary, cat):
        self.semantic_dir = semantic_dir

        files_list = open(os.path.join(semantic_dir, 'semantic3d_test.txt'), 'r')
        all_files = files_list.read().split('\n')
        all_files = [i for i in all_files if i != '']
        files_list.close()
        self.num_files = len(all_files)

        self.semi = semi
        self.pc_size = pc_size
        self.binary = binary
        self.cat = cat

        self.pc_size_dir = str(self.pc_size)
        if not self.semi:
            self.learn_dir = 'full_sup'
        else:
            self.learn_dir = 'semi_sup'
        if not binary:
            self.bin_dir = 'all'
            self.cat_dir = ''
        else:
            self.bin_dir = 'binary'
            self.cat_dir = str(cat)

        self.master_dir = os.path.join(self.semantic_dir, 'torch_data',
                                       self.pc_size_dir, self.learn_dir, self.bin_dir, self.cat_dir)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """

        pc = torch.load(os.path.join(self.master_dir, str(index) + '.pt'))

        if not self.semi:
            labels = torch.load(os.path.join(self.master_dir, str(index) + '_labels.pt'))
            return pc, labels
        else:
            return pc, pc[:, 3:]

        # # For overfitting:
        # pc = torch.load(os.path.join(self.semantic_dir, 'torch_data', '7.pt'))
        # return pc, pc[:, 3:]

    def __len__(self):
        return self.num_files

        # # For overfitting:
        # return 2