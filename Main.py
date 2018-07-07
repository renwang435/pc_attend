import logging
import time
import sys

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data

from CoreNet import CoreNet
from GlimpseNet import GlimpseNet
from LocNet import LocNet
from Params import Params
from Utils import *

logging.getLogger().setLevel(logging.INFO)

mnist = input_data.read_data_sets('MNIST_data', one_hot=False)

params = Params()
n_steps = params.step

save_dir = "chckPts/"
save_prefix = "save"

draw = 1
animate = 1

# Helper function 1/2 for visualization
def toMnistCoordinates(coordinate_tanh, img_size):
    '''
    Transform coordinate in [-1,1] to mnist
    :param coordinate_tanh: vector in [-1,1] x [-1,1]
    :return: vector in the corresponding mnist coordinate
    '''
    return np.round(((coordinate_tanh + 1) / 2.0) * img_size)

# Helper function 2/2 for visualization
def plotWholeImg(img, img_size, sampled_locs_fetched):
    plt.imshow(np.reshape(img, [img_size, img_size]),
               cmap=plt.get_cmap('gray'), interpolation="nearest")

    plt.ylim((img_size - 1, 0))
    plt.xlim((0, img_size - 1))

    # transform the coordinate to mnist map
    sampled_locs_mnist_fetched = toMnistCoordinates(sampled_locs_fetched, img_size)
    # visualize the trace of successive nGlimpses (note that x and y coordinates are "flipped")
    plt.plot(sampled_locs_mnist_fetched[0, :, 1], sampled_locs_mnist_fetched[0, :, 0], '-o',
             color='lawngreen')
    plt.plot(sampled_locs_mnist_fetched[0, -1, 1], sampled_locs_mnist_fetched[0, -1, 0], 'o',
             color='red')

# placeholders
images_ph = tf.placeholder(tf.float32,
                           [None, params.original_size * params.original_size *
                            params.num_channels])
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
sampled_locs = tf.reshape(sampled_locs, (params.num_glimpses, params.batch_size, 2))
sampled_locs = tf.transpose(sampled_locs, [1, 0, 2])

glimpse_images = tf.concat(axis=0, values=glimpse_images)


# print(sampled_locs)
# sys.exit(1)

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
max_p_y = tf.arg_max(softmax, 1)
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
training_steps_per_epoch = mnist.train.num_examples // params.batch_size
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
    # epoch = 0
    for i in range(n_steps):
        images, labels = mnist.train.next_batch(params.batch_size)
        # duplicate M times, see eqn (2)
        # images = np.tile(images, [params.M, 1])
        # labels = np.tile(labels, [params.M])
        loc_net.samping = True

        glimpse_images_fetched, sampled_locs_fetched, pred, true, adv_val, baselines_mse_val, xent_val, \
        logllratio_val, reward_val, loss_val, lr_val, _ = sess.run(
                [glimpse_images, sampled_loc_arr, max_p_y, correct_y, advs, baselines_mse, xent, logllratio,
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
                fig = plt.figure(1)
                txt = fig.suptitle("-", fontsize=36, fontweight='bold')
                plt.ion()
                plt.show()
                plt.subplots_adjust(top=0.7)
                plotImgs = []
                img_size = params.original_size

                f_glimpse_images = np.reshape(glimpse_images_fetched, \
                                              (params.num_glimpses,
                                               params.batch_size, params.depth,
                                               params.bandwidth, params.bandwidth))

                if animate:
                    fillList = False
                    if len(plotImgs) == 0:
                        fillList = True

                    # display the first image in the in mini-batch
                    nCols = params.depth + 1
                    plt.subplot2grid((params.depth, nCols), (0, 1), rowspan=params.depth, colspan=params.depth)

                    # display the entire image
                    sampled_locs_fetched = np.transpose(sampled_locs_fetched, (1, 0, 2))
                    plotWholeImg(images[0, :], img_size, sampled_locs_fetched)

                    # display the glimpses
                    for y in range(params.num_glimpses):
                        txt.set_text('Step: %.6d \nPrediction: %i -- Truth: %i\nGlimpse: %i/%i'
                                     % (i, pred[0], true[0], (y + 1),
                                        params.num_glimpses))

                        for x in range(params.depth):
                            plt.subplot(params.depth, nCols, 1 + nCols * x)
                            if fillList:
                                plotImg = plt.imshow(f_glimpse_images[y, 0, x], cmap=plt.get_cmap('gray'),
                                                     interpolation="nearest")
                                plotImg.autoscale()
                                plotImgs.append(plotImg)
                            else:
                                plotImgs[x].set_data(f_glimpse_images[y, 0, x])
                                plotImgs[x].autoscale()
                        fillList = False

                        # fig.canvas.draw()
                        time.sleep(0.1)
                        plt.pause(0.00005)

                else:
                    txt.set_text(
                        'PREDICTION: %i\nTRUTH: %i' % (pred[0], true[0]))
                    for x in range(params.depth):
                        for y in range(params.num_glimpses):
                            plt.subplot(params.depth, params.num_glimpses, x * params.num_glimpses + y + 1)
                            plt.imshow(f_glimpse_images[y, 0, x], cmap=plt.get_cmap('gray'), interpolation="nearest")

                    plt.draw()
                    time.sleep(0.05)
                    plt.pause(0.0001)


        if i and i % training_steps_per_epoch == 0:
            # Save model
            logging.info('Saving model to ' + save_dir)
            saver.save(sess, save_dir + save_prefix + str(i) + ".ckpt")

            # Evaluation
            for dataset in [mnist.validation, mnist.test]:
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
                    #                    [params.M, -1, params.num_classes])
                    # softmax_val = np.mean(softmax_val, 0)
                    pred_labels_val = np.argmax(softmax_val, 1)
                    pred_labels_val = pred_labels_val.flatten()
                    correct_cnt += np.sum(pred_labels_val == labels_bak)
                acc = correct_cnt / num_samples
                if dataset == mnist.validation:
                    logging.info('valid accuracy = {}'.format(acc))
                else:
                    logging.info('test accuracy = {}'.format(acc))