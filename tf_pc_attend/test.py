import glob2
import re
import numpy as np
import sys
import time

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


data_dir = ('modelnet40_normal_resampled')

# Helper function for visualization
def plotWholePC(pc, sampled_locs_fetched):
    pc = np.reshape(pc, (num_points_per_pc, loc_dim))
    xs = pc[:, 0]
    ys = pc[:, 1]
    zs = pc[:, 2]
    
    ax0.scatter(xs, ys, zs)

    # # visualize the trace of successive nGlimpses (note that x and y coordinates are "flipped")
    ax0.plot(sampled_locs_fetched[0, :, 0], sampled_locs_fetched[0, :, 1], sampled_locs_fetched[0, :, 2], 
            '-o', color='lawngreen')
    
    last_loc_movement = sampled_locs_fetched[0, -2:]
    ax0.plot(last_loc_movement[:, 0], last_loc_movement[:, 1], last_loc_movement[:, 2], 
            '->', color='red')

# Helper function for visualization
def plotGlimpses(f_glimpse_images, glimpse_axes):
    for i in range(len(f_glimpse_images)):
        xs = f_glimpse_images[i, 0, :, 0]
        ys = f_glimpse_images[i, 0, :, 1]
        zs = f_glimpse_images[i, 0, :, 2]
    
        glimpse_axes[i].scatter(xs, ys, zs)

if __name__ == '__main__':
    glimpse_images_fetched = np.load('glimpse_images.npy')
    sampled_locs_fetched = np.load('sampled_locs.npy')
    image = np.load('image.npy')

    num_glimpses = 6
    batch_size = 32
    num_points_per_pc = 10000
    num_points_per_glimpse = 100
    loc_dim = 3

    i = 0
    while True:
        i += 1
        if i and i % 100 == 0:
            # Now we do some visualization
            grid = plt.GridSpec(num_glimpses * 2, num_glimpses, wspace=0.4, hspace=0.3)
            ax0 = plt.subplot(grid[0:num_glimpses, :], projection='3d')
            glimpse_axes = []
            for j in range(num_glimpses):
                curr_ax = plt.subplot(grid[num_glimpses:, j], projection='3d')
                glimpse_axes.append(curr_ax)
            
            txt = plt.suptitle("-", fontsize=36, fontweight='bold')
            plt.ion()
            # plt.show()
            plt.subplots_adjust(top=0.7)

            f_glimpse_images = np.reshape(glimpse_images_fetched,
                                            (num_glimpses, batch_size,
                                            num_points_per_glimpse, loc_dim))

            # display the entire image along with the sampled locations
            sampled_locs_fetched = np.transpose(sampled_locs_fetched, (1, 0, 2))
            plotWholePC(image, sampled_locs_fetched)

            # Display the glimpses
            plotGlimpses(f_glimpse_images, glimpse_axes)

            plt.draw()
            time.sleep(0.05)
            # plt.pause(0.0001)
            plt.pause(2)
            # plt.close()

