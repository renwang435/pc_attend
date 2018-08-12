import logging
import time
import sys
import platform

import numpy as np
import matplotlib
if (platform.system() == 'Darwin'):
    matplotlib.use('TkAgg')     # For visualization on OS X

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


from tensorflow.examples.tutorials.mnist import input_data

from CoreNet import CoreNet
from GlimpseNet import GlimpseNet
from LocNet import LocNet
from Params import Params
from Utils import *

logging.getLogger().setLevel(logging.INFO)

params = Params()
data_dir = ('modelnet40_normal_resampled')
pc_dataset = read_data_sets(data_dir, params.num_points_per_pc)

label_list = open(data_dir + '/modelnet40_shape_names.txt', 'r')
all_labels = label_list.read().split('\n')
all_labels = [i for i in all_labels if i != '']
label_list.close()

n_steps = params.step

save_dir = "chckPts/"
save_prefix = "save"

draw = 1
display = 1
save = 1
pause_time = 2 # Useful for debugging, set higher to allow more time to interact with plots

# Helper function for visualization
def plotWholePC(pc, sampled_locs_fetched):
    pc = np.reshape(pc, (params.num_points_per_pc, params.loc_dim))
    xs = pc[:, 0]
    ys = pc[:, 1]
    zs = pc[:, 2]
    
    ax0.scatter(xs, ys, zs)

    # # visualize the trace of successive nGlimpses (note that x and y coordinates are "flipped")
    ax0.plot(sampled_locs_fetched[0, :, 0], sampled_locs_fetched[0, :, 1], sampled_locs_fetched[0, :, 2], 
            '-o', color='lawngreen')
    
    last_loc_movement = sampled_locs_fetched[0, -2:]
    ax0.plot(last_loc_movement[:, 0], last_loc_movement[:, 1], last_loc_movement[:, 2], 
            '-o', color='red')

# Helper function for visualization
def plotGlimpses(f_glimpse_images, glimpse_axes):
    for i in range(len(f_glimpse_images)):
        xs = f_glimpse_images[i, 0, :, 0]
        ys = f_glimpse_images[i, 0, :, 1]
        zs = f_glimpse_images[i, 0, :, 2]
    
        glimpse_axes[i].scatter(xs, ys, zs)

# placeholders
images_ph = tf.placeholder(tf.float32,
                        [None, params.num_points_per_pc * params.loc_dim])
labels_ph = tf.placeholder(tf.int64, [None])

#Build our Glimpse and Location networks
with tf.variable_scope('glimpse_net'):
    gl = GlimpseNet(params, images_ph)
with tf.variable_scope('loc_net'):
    loc_net = LocNet(params)

#Define the number of examples we have
num_examples = tf.shape(images_ph)[0]

#Build our Core Network
with tf.variable_scope('core_net'):
    core_net = CoreNet(params, gl, loc_net)
    outputs, sampled_loc_arr, glimpse_images = core_net(num_examples)

# For visualization purposes
sampled_locs = tf.concat(axis=0, values=sampled_loc_arr)
sampled_locs = tf.reshape(sampled_locs, (params.num_glimpses, params.batch_size, params.loc_dim))
sampled_locs = tf.transpose(sampled_locs, [1, 0, 2])

glimpse_images = tf.concat(axis=0, values=glimpse_images)


# Define time-independent baselines for variance reduction (as per eqn (2) in the paper)
with tf.variable_scope('baseline'):
    w_baseline = weight_variable((params.cell_output_size, 1))
    b_baseline = bias_variable((1,))
baselines = []

for _, output in enumerate(outputs[1:]):
    baseline_t = tf.nn.xw_plus_b(output, w_baseline, b_baseline)
    baseline_t = tf.squeeze(baseline_t)
    baselines.append(baseline_t)

baselines = tf.stack(baselines)  # [timesteps, batch_size]
baselines = tf.transpose(baselines)  # [batch_size, timesteps]

# Classification task uses only the output at the last timestep
output = outputs[-1]

# Build action/classification network.
with tf.variable_scope('cls'):
    w_logit = weight_variable((params.cell_output_size, params.num_classes))
    b_logit = bias_variable((params.num_classes,))
logits = tf.nn.xw_plus_b(output, w_logit, b_logit)
softmax = tf.nn.softmax(logits)
max_p_y = tf.argmax(softmax, axis=1)
correct_y = tf.cast(labels_ph, tf.int64)

# Define our cross entropy loss first for the classification task
xent = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels_ph)
xent = tf.reduce_mean(xent)
pred_labels = tf.argmax(logits, 1)

# 0/1 reward (0 if the object is classified incorrectly, 1 otherwise)
reward = tf.cast(tf.equal(pred_labels, labels_ph), tf.float32)
rewards = tf.expand_dims(reward, 1)  # [batch_size, 1]
rewards = tf.tile(rewards, (1, params.num_glimpses))  # [batch_sz, timesteps]
logll = loglikelihood(core_net.loc_mean_arr, core_net.sampled_loc_arr, params.loc_std)
advs = rewards - tf.stop_gradient(baselines)
logllratio = tf.reduce_mean(logll * advs)
reward = tf.reduce_mean(reward)

baselines_mse = tf.reduce_mean(tf.square((rewards - baselines)))
var_list = tf.trainable_variables()

# Hybrid loss for classification task
loss = -logllratio + xent + baselines_mse
grads = tf.gradients(loss, var_list)
grads, _ = tf.clip_by_global_norm(grads, params.max_grad_norm)

# learning rate
global_step = tf.get_variable(
    'global_step', [], initializer=tf.constant_initializer(0), trainable=False)
training_steps_per_epoch = pc_dataset.train.num_examples // params.batch_size
starter_learning_rate = params.lr_start
# decay per training epoch
learning_rate = tf.train.exponential_decay(
    starter_learning_rate,
    global_step,
    training_steps_per_epoch,
    0.97,
    staircase=True)
learning_rate = tf.maximum(learning_rate, params.lr_min)
opt = tf.train.AdamOptimizer(learning_rate)
train_op = opt.apply_gradients(zip(grads, var_list), global_step=global_step)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver(max_to_keep=99999999)
    for i in range(n_steps):
        images, labels = pc_dataset.train.next_batch(params.batch_size)
        # duplicate M times, see eqn (2)
        # images = np.tile(images, [params.M, 1])
        # labels = np.tile(labels, [params.M])
        loc_net.sampling = True

        (glimpse_images_fetched, sampled_locs_fetched, pred, true, 
            adv_val, baselines_mse_val, xent_val, logllratio_val,
            reward_val, loss_val, lr_val, _) = sess.run(
                [glimpse_images, sampled_loc_arr, max_p_y, correct_y, 
                advs, baselines_mse, xent, logllratio,
                reward, loss, learning_rate, train_op],
                feed_dict={
                    images_ph: images,
                    labels_ph: labels
                })
        
        if i and i % 100 == 0:
            logging.info('step {}: lr = {:3.6f}'.format(i, lr_val))
            logging.info(
                'step {}: reward = {:3.4f}\tloss = {:3.4f}\txent = {:3.4f}'.format(
                    i, reward_val, loss_val, xent_val))
            logging.info('llratio = {:3.4f}\tbaselines_mse = {:3.4f}'.format(
                logllratio_val, baselines_mse_val))

            # Now we do some visualization
            if draw:
                grid = plt.GridSpec(params.num_glimpses * 2, params.num_glimpses, wspace=0.4, hspace=0.4)
                ax0 = plt.subplot(grid[0:params.num_glimpses, :], projection='3d')
                glimpse_axes = []
                for j in range(params.num_glimpses):
                    curr_ax = plt.subplot(grid[params.num_glimpses:, j], projection='3d')
                    glimpse_axes.append(curr_ax)
                
                txt = plt.suptitle("-", fontsize=36, fontweight='bold')
                pred_text = str(all_labels[pred[0]])
                true_text = str(all_labels[true[0]])
                txt.set_text('PREDICTION: %s\nTRUTH: %s' % (pred_text, true_text))

                plt.subplots_adjust(top=0.7)

                f_glimpse_images = np.reshape(glimpse_images_fetched,
                                                (params.num_glimpses, params.batch_size,
                                                params.num_points_per_glimpse, params.loc_dim))

                # display the entire image along with the sampled locations
                sampled_locs_fetched = np.transpose(sampled_locs_fetched, (1, 0, 2))
                plotWholePC(images[0, :], sampled_locs_fetched)

                # Display the glimpses
                plotGlimpses(f_glimpse_images, glimpse_axes)

                if display:
                    plt.ion()
                    plt.draw()
                    time.sleep(0.05)
                    plt.pause(pause_time)
                elif save:
                    plt.savefig('%s_%s_%s.png' % (str(i), pred_text, true_text))
                    plt.close()
                else:
                    pass           

        if i and i % training_steps_per_epoch == 0:
            # Save model
            logging.info('Saving model to ' + save_dir)
            saver.save(sess, save_dir + save_prefix + str(i) + ".ckpt")

            # Evaluation
            for dataset in [pc_dataset.validation, pc_dataset.test]:
                steps_per_epoch = dataset.num_examples // params.eval_batch_size
                correct_cnt = 0
                num_samples = steps_per_epoch * params.batch_size
                loc_net.sampling = True
                for test_step in range(steps_per_epoch):
                    images, labels = dataset.next_batch(params.batch_size)
                    labels_bak = labels
                    # Duplicate M times
                    # images = np.tile(images, [params.M, 1])
                    # labels = np.tile(labels, [params.M])
                    softmax_val = sess.run(softmax,
                                    feed_dict={
                                        images_ph: images,
                                        labels_ph: labels
                                    })
                    # softmax_val = np.reshape(softmax_val,
                    #                 [params.M, -1, params.num_classes])
                    # softmax_val = np.mean(softmax_val, 0)
                    pred_labels_val = np.argmax(softmax_val, 1)
                    pred_labels_val = pred_labels_val.flatten()
                    correct_cnt += np.sum(pred_labels_val == labels_bak)
                acc = correct_cnt / num_samples
                if dataset == pc_dataset.validation:
                    logging.info('valid accuracy = {}'.format(acc))
                else:
                    logging.info('test accuracy = {}'.format(acc))
