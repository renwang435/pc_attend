import numpy as np

import torch
from PointCloud import PointCloud
from torch.utils.data.sampler import SubsetRandomSampler

def get_train_valid_loader(semantic_dir,
                           random_seed=42,
                           batch_size=1,
                           valid_size=0.34,
                           shuffle=True,
                           semi=True,
                           pc_size=500000,
                           binary=True,
                           cat=0,
                           num_workers=4,
                           pin_memory=False):
    error_msg = "[!] valid_size should be in the range [0, 1]."
    assert ((valid_size >= 0) and (valid_size <= 1)), error_msg

    # load dataset
    dataset = PointCloud(semantic_dir, semi, pc_size, binary, cat)

    num_train = len(dataset)
    indices = list(range(num_train))
    split = int(np.floor(valid_size * num_train))

    if shuffle:
        np.random.seed(random_seed)
        np.random.shuffle(indices)

    train_idx, valid_idx = indices[split:], indices[:split]

    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    train_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, sampler=train_sampler,
        num_workers=num_workers, pin_memory=pin_memory,
    )

    valid_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, sampler=valid_sampler,
        num_workers=num_workers, pin_memory=pin_memory,
    )

    return (train_loader, valid_loader)