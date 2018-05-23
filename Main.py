import logging

import numpy as np
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
    outputs = core_net(num_examples)

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
    for i in range(n_steps):
        images, labels = mnist.train.next_batch(params.batch_size)
        # duplicate M times, see eqn (2)
        images = np.tile(images, [params.M, 1])
        labels = np.tile(labels, [params.M])
        loc_net.samping = True
        adv_val, baselines_mse_val, xent_val, logllratio_val, \
            reward_val, loss_val, lr_val, _ = sess.run(
                [advs, baselines_mse, xent, logllratio,
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

        if i and i % training_steps_per_epoch == 0:
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
                    images = np.tile(images, [params.M, 1])
                    labels = np.tile(labels, [params.M])
                    softmax_val = sess.run(softmax,
                                     feed_dict={
                                         images_ph: images,
                                         labels_ph: labels
                                     })
                    softmax_val = np.reshape(softmax_val,
                                       [params.M, -1, params.num_classes])
                    softmax_val = np.mean(softmax_val, 0)
                    pred_labels_val = np.argmax(softmax_val, 1)
                    pred_labels_val = pred_labels_val.flatten()
                    correct_cnt += np.sum(pred_labels_val == labels_bak)
                acc = correct_cnt / num_samples
                if dataset == mnist.validation:
                    logging.info('valid accuracy = {}'.format(acc))
                else:
                    logging.info('test accuracy = {}'.format(acc))