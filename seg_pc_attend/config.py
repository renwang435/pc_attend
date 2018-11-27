import argparse

arg_lists = []
parser = argparse.ArgumentParser(description='RAM')


def str2bool(v):
    return v.lower() in ('true', '1')


def add_argument_group(name):
    arg = parser.add_argument_group(name)
    arg_lists.append(arg)
    return arg

# data params
data_arg = add_argument_group('Data Params')
data_arg.add_argument('--valid_size', type=float, default=0.34,
                      help='Proportion of training set used for validation')
data_arg.add_argument('--mask_rate', type=float, default=0.4,
                      help='Masking rate for deep sets modules')
data_arg.add_argument('--batch_size', type=int, default=1,
                      help='# of images in each batch of data')
data_arg.add_argument('--num_workers', type=int, default=4,
                      help='# of subprocesses to use for data loading')
data_arg.add_argument('--shuffle', type=str2bool, default=True,
                      help='Whether to shuffle the train and valid indices')
data_arg.add_argument('--semi', type=bool, default=True,
                      help='Semi supervised (true) or fully supervised (false) learning')
data_arg.add_argument('--pc_size', type=int, default=1000,
                      help='Sampling density of point clouds')
data_arg.add_argument('--binary', type=bool, default=True,
                      help='Binary segmentation or full segmentation')
data_arg.add_argument('--cat', type=int, default=0,
                      help='Foreground category (for binary segmentation)')

# training params
train_arg = add_argument_group('Training Params')
train_arg.add_argument('--is_train', type=str2bool, default=True,
                       help='Whether to train or test the model')
train_arg.add_argument('--momentum', type=float, default=0.5,
                       help='Nesterov momentum value')
train_arg.add_argument('--epochs', type=int, default=150,
                       help='# of epochs to train for')
train_arg.add_argument('--init_lr', type=float, default=2e-4,
                       help='Initial learning rate value')
train_arg.add_argument('--lr_patience', type=int, default=10,
                       help='Number of epochs to wait before reducing lr')
train_arg.add_argument('--train_patience', type=int, default=25,
                       help='Number of epochs to wait before stopping train')

# other params
misc_arg = add_argument_group('Misc.')
misc_arg.add_argument('--use_gpu', type=str2bool, default=False,
                      help="Whether to run on the GPU")
misc_arg.add_argument('--best', type=str2bool, default=True,
                      help='Load best model or most recent for testing')
misc_arg.add_argument('--random_seed', type=int, default=42,
                      help='Seed to ensure reproducibility')
misc_arg.add_argument('--data_dir', type=str, default='./data/semantic3d',
                      help='Directory in which data is stored')
misc_arg.add_argument('--ckpt_dir', type=str, default='./ckpt',
                      help='Directory in which to save model checkpoints')
misc_arg.add_argument('--logs_dir', type=str, default='./logs/',
                      help='Directory in which Tensorboard logs wil be stored')
misc_arg.add_argument('--resume', type=str2bool, default=False,
                      help='Whether to resume training from checkpoint')

def get_config():
    config, unparsed = parser.parse_known_args()
    return config, unparsed
