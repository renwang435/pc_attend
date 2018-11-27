import numpy as np
import os
import torch
import shutil
import sys
from config import get_config

def save_all_tensors(semantic_dir, sampling, one_hot_labels, all_binary_labels, data, i):
    learning_types = ['full_sup', 'semi_sup']

    # Write tensors for 1-of-8 problem, fully supervised
    write_dir = os.path.join(semantic_dir, str(sampling), learning_types[0], 'all')
    torch.save(torch.from_numpy(data), os.path.join(write_dir, str(i) + '.pt'))
    torch.save(torch.from_numpy(one_hot_labels), os.path.join(write_dir, str(i) + '_labels.pt'))

    # Write tensors for 1-of-8 problem, semi supervised
    write_dir = os.path.join(semantic_dir, str(sampling), learning_types[1], 'all')
    final_one_hot_tensor = torch.from_numpy(np.hstack((data, one_hot_labels)))
    torch.save(final_one_hot_tensor, os.path.join(write_dir, str(i) + '.pt'))

    for j in range(8):
        # Write tensors for binary problem, fully supervised
        write_dir = os.path.join(semantic_dir, str(sampling), learning_types[0], 'binary', str(j))
        torch.save(torch.from_numpy(data), os.path.join(write_dir, str(i) + '.pt'))
        torch.save(torch.from_numpy(all_binary_labels[j]), os.path.join(write_dir, str(i) + '_labels.pt'))

        # Write tensors for binary problem, semi supervised
        write_dir = os.path.join(semantic_dir, str(sampling), learning_types[1], 'binary', str(j))
        final_binary_tensor = torch.from_numpy(np.hstack((data, all_binary_labels[j])))
        torch.save(final_binary_tensor, os.path.join(write_dir, str(i) + '.pt'))

def text_labels_to_torch(path_to_file, sampling):
    file = open(path_to_file, 'r')
    all_points = file.read().split('\n')
    file.close()
    points = [int(i) - 1 for i in all_points if i != '']

    point_cloud = np.array(points)
    ret_indices = np.where(point_cloud >= 0)
    point_cloud = point_cloud[ret_indices]

    # Loop until the requisite number of points is obtained
    total_num_points_left = sampling
    sampled_pc = np.array([]).astype(int)
    all_permute_indices = []
    while (total_num_points_left > 0):
        if (total_num_points_left < len(point_cloud)):
            permute_indices = np.random.RandomState(seed=42).permutation(len(point_cloud))[:total_num_points_left]
            sampled_pc = np.hstack((sampled_pc, point_cloud[permute_indices]))
            total_num_points_left -= total_num_points_left
            all_permute_indices.append(permute_indices)
        else:
            permute_indices = np.random.RandomState(seed=42).permutation(len(point_cloud))
            sampled_pc = np.hstack((sampled_pc, point_cloud[permute_indices]))
            total_num_points_left -= len(point_cloud)
            all_permute_indices.append(permute_indices)
    point_cloud = sampled_pc

    # One hot encode labels (1 of 8 problem)
    possible_labels = np.eye(8)
    one_hot_labels = np.array([possible_labels[i] for i in point_cloud]).astype(int)

    # Binarize labels (binary problem)
    # Need to binarize wrt every class
    all_binary_labels = []
    for i in range(8):
        curr_binary_labels = np.empty((len(point_cloud), 2)).astype(int)
        for j in range(len(curr_binary_labels)):
            if (point_cloud[j] == i):
                curr_binary_labels[j] = [1, 0]
            else:
                curr_binary_labels[j] = [0, 1]
        all_binary_labels.append(curr_binary_labels)

    return one_hot_labels, all_binary_labels, ret_indices, np.concatenate(all_permute_indices).flatten()

def text_train_to_torch(path_to_file, keep_indices, permute_indices):
    file = open(path_to_file, 'r')
    all_points = file.read().split('\n')
    file.close()
    points = [i for i in all_points if i != '']
    points = [points[i] for i in keep_indices[0]]
    points = [points[i] for i in permute_indices]

    xs = []
    ys = []
    zs = []
    for point in points:
        points_sep = point.split(' ')
        x = float(points_sep[0])
        y = float(points_sep[1])
        z = float(points_sep[2])

        xs.append(x)
        ys.append(y)
        zs.append(z)

    point_cloud = np.vstack((np.array(xs).T, np.array(ys).T, np.array(zs).T)).T
    max_element = max(abs(i) for i in point_cloud)
    point_cloud = point_cloud / max_element

    return point_cloud

def write_torch_dataset(semantic_dir, all_sampling_levels):
    print('Loading Semantic3D files...')
    # Read in the Semantic3D labels
    files_list = open(semantic_dir + '/semantic3d_test.txt', 'r')
    all_files = files_list.read().split('\n')
    all_files = [i for i in all_files if i != '']
    files_list.close()

    # Write the training and label PyTorch tensors
    print('Writing PyTorch tensors...')

    i = 0
    total = len(all_files)
    for file in all_files:
        for sampling in all_sampling_levels:

            # Write labels files
            path_to_file = os.path.join(semantic_dir, 'sem8_labels_training', file + '.labels')
            one_hot_labels, all_binary_labels, ret_indices, permute_indices = text_labels_to_torch(path_to_file, sampling)

            # Write training tensors
            path_to_file = os.path.join(semantic_dir, file + '.txt')
            data = text_train_to_torch(path_to_file, ret_indices, permute_indices)

            # Save in three separate folders, with each folder containing a separate set of folders for diff sampling rates
            save_master_dir = os.path.join(semantic_dir, 'torch_data')
            save_all_tensors(save_master_dir, sampling, one_hot_labels, all_binary_labels, data, i)

        i += 1
        print('Completed: %.2f %%' % (i * 100 / total))

def prepare_dirs(config, all_sampling_levels, super_dir):
    for path in [config.ckpt_dir, config.logs_dir]:
        if not os.path.exists(path):
            os.makedirs(path)

    if (os.path.exists(super_dir)):
        print('Data path already exists...')
        sys.exit(0)

    os.makedirs(super_dir)

    learning_types = ['full_sup', 'semi_sup']

    for sampling in all_sampling_levels:
        master_dir = os.path.join(super_dir, str(sampling))
        if not os.path.exists(master_dir):
            os.makedirs(master_dir)

        for learning_type in learning_types:
            bin_dir = os.path.join(master_dir, learning_type, 'binary')
            if not os.path.exists(bin_dir):
                for i in range(8):
                    os.makedirs(os.path.join(bin_dir, str(i)))
            all_dir = os.path.join(master_dir, learning_type, 'all')
            if not os.path.exists(all_dir):
                os.makedirs(all_dir)

def rmdir_if_exists(input_dir):
    if (os.path.exists(input_dir)):
        shutil.rmtree(input_dir)

def data_prep(config):
    torch_data_dir = os.path.join(config.data_dir, 'torch_data')
    # rmdir_if_exists(torch_data_dir)

    all_sampling_levels = np.array([1e5, 2.5e5, 5e5, 7.5e5,
                                    1e6, 2e6, 2.5e6, 5e6, 7.5e6,
                                    1e7]).astype(int)
    # all_sampling_levels = np.array([1e2, 1e3, 1e4,
    #                                 1e5, 5e5]).astype(int)

    prepare_dirs(config, all_sampling_levels, torch_data_dir)
    write_torch_dataset(config.data_dir, all_sampling_levels)

if __name__ == '__main__':
    config, unparsed = get_config()
    data_prep(config)
