import math
import os
import sys
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

master_dir = 'modelnet40_normal_resampled'

if __name__ == '__main__':
    file = open(os.path.join(master_dir, 'airplane', 'airplane_0273.txt'), 'r')
    all_points = file.read().split('\n')
    points = [i for i in all_points if i != '']

    dists = []
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
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(xs, ys, zs)
    plt.show()
