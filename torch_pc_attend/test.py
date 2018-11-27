import os
import sys
import pickle
import shutil
import time

import torch
import torch.nn.functional as F
import torch.optim as optim
from tensorboard_logger import configure, log_value
from torch.autograd import Variable
from tqdm import tqdm

from model import RecurrentAttention
from utils import AverageMeter

from trainer import Trainer
from config import get_config
from utils import prepare_dirs, save_config, write_torch_dataset
from data_loader import get_test_loader, get_train_valid_loader

if __name__ == '__main__':
    with open('partition_dict.txt', 'rb') as fp:
        d = pickle.load(fp)

    print(len(d['train']))
    print(len(d['test']))