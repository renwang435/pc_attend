import json
import os

import torch

from Trainer import Trainer
from config import get_config
from data_loader import get_train_valid_loader


def save_config(config):
    if config.semi:
        str1 = 'semi'
    else:
        str1 = 'fully'
    if config.binary:
        str2 = 'binary'
        str3 = str(config.cat)
    else:
        str2 = 'all'
        str3 = 'nocat'
    str4 = config.pc_size

    model_name = 'dsseg_{}_{}_{}_{}_{}'.format(
        config.init_lr, str1, str2, str3, str4
    )
    filename = model_name + '_params.json'
    param_path = os.path.join(config.ckpt_dir, filename)

    print("[*] Model Checkpoint Dir: {}".format(config.ckpt_dir))
    print("[*] Param Path: {}".format(param_path))

    with open(param_path, 'w') as fp:
        json.dump(config.__dict__, fp, indent=4, sort_keys=True)


def main(config):
    # ensure reproducibility
    torch.manual_seed(config.random_seed)
    kwargs = {}
    if config.use_gpu:
        torch.cuda.manual_seed(config.random_seed)
        kwargs = {'num_workers': 1, 'pin_memory': True}

    data_loader = get_train_valid_loader(
        config.data_dir,
        config.random_seed, config.batch_size,
        config.valid_size, config.shuffle,
        config.semi, config.pc_size, config.binary, config.cat,
        **kwargs
    )

    trainer = Trainer(config, data_loader)

    save_config(config)
    trainer.train()

    trainer.test()


if __name__ == '__main__':
    config, unparsed = get_config()
    main(config)