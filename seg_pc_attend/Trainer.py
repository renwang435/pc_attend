from __future__ import division

import os
import shutil
import time

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from tqdm import tqdm
from model import RecurrentAttention

from deep_sets import DTanh

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class Trainer(object):
    def __init__(self, config, data_loader):
        self.config = config

        self.train_loader = data_loader[0]
        self.valid_loader = data_loader[1]
        self.num_train = len(self.train_loader.sampler.indices)
        self.num_valid = len(self.valid_loader.sampler.indices)

        if self.config.binary:
            self.num_classes = 2
            self.loss = F.binary_cross_entropy_with_logits
            namestr2 = 'binary'
            namestr3 = str(config.cat)
        else:
            self.num_classes = 8
            self.loss = F.cross_entropy
            namestr2 = 'all'
            namestr3 = 'nocat'

        # model params
        if self.config.semi:
            namestr1 = 'semi'
            self.input_dim = self.num_classes + 3
        else:
            namestr1 = 'fully'
            self.input_dim = 3
        self.output_dim = self.num_classes
        self.mask_rate = config.mask_rate
        self.pc_size = config.pc_size

        # training params
        self.epochs = config.epochs
        self.start_epoch = 0
        self.lr = config.init_lr

        # misc params
        self.use_gpu = config.use_gpu
        self.ckpt_dir = config.ckpt_dir
        self.best = config.best
        self.best_mIoU = -10
        self.best_acc = 0
        self.counter = 0
        self.lr_patience = config.lr_patience
        self.train_patience = config.train_patience
        self.resume = config.resume
        self.model_name = 'dsseg_{}_{}_{}_{}_{}'.format(
            config.init_lr, namestr1, namestr2, namestr3, config.pc_size
        )

        # attention parameters
        # glimpse params
        # glimpse network params
        self.num_points_per_pc = config.pc_size
        self.num_points_per_sample = config.num_points_per_sample
        self.box_size = config.box_size
        self.glimpse_scale = config.glimpse_scale
        self.num_samples = config.num_samples
        self.loc_hidden = config.loc_hidden
        self.glimpse_hidden = config.glimpse_hidden

        # core network params
        self.num_glimpses = config.num_glimpses
        self.hidden_size = config.hidden_size

        # reinforce params
        self.std = config.std
        self.M = config.M

        # # build DS model
        # self.model = DTanh(
        #     self.input_dim, self.output_dim
        # )
        # build RAM model
        self.model = RecurrentAttention(
            self.num_points_per_pc, self.num_points_per_sample, self.num_samples, self.box_size,
            self.glimpse_scale, self.num_channels, self.loc_hidden, self.glimpse_hidden,
            self.std, self.hidden_size, self.num_classes, self.use_gpu
        )
        if self.use_gpu:
            self.model.cuda()

        print('[*] Number of model parameters: {:,}'.format(
            sum([p.data.nelement() for p in self.model.parameters()])))

        self.optimizer = optim.Adam(
            self.model.parameters(), lr=self.lr,
        )

    def mask_tensor(self, x, rate):
        """
        Masks a percentage of the entries in tensor x randomly
        """

        tensor_len = x.shape[1]
        if (rate == 0.):
            return x, np.arange(tensor_len)

        num_index = int(rate * tensor_len)
        permute_indices = np.random.RandomState(seed=42).permutation(tensor_len)[:num_index]
        zero_mask = torch.zeros(x.shape[-1] - 3, dtype=torch.float32)
        if self.use_gpu:
            zero_mask = zero_mask.cuda()
        x[:, permute_indices, 3:] = zero_mask

        return x, permute_indices

    def train(self):
        """
        Train the model on the training set.

        A checkpoint of the model is saved after each epoch
        and if the validation accuracy is improved upon,
        a separate ckpt is created for use on the test set.
        """
        # load the most recent checkpoint
        if self.resume:
            self.load_checkpoint(best=False)

        print("\n[*] Train on {} samples, validate on {} samples".format(
            self.num_train, self.num_valid)
        )

        if self.config.binary:
            for epoch in range(self.start_epoch, self.epochs):

                print(
                    '\nEpoch: {}/{} - LR: {:.6f}'.format(
                        epoch+1, self.epochs, self.lr)
                )

                # train for 1 epoch
                # train_loss, train_acc = self.train_one_epoch(epoch)
                train_loss, train_acc, zeros_acc, ones_acc = self.train_one_epoch(epoch)

                # evaluate on validation set
                # valid_loss, valid_acc = self.validate(epoch)
                valid_loss, valid_acc, val_zeros_acc, val_ones_acc = self.validate(epoch)

                # mIoU = (np.mean(valid_IoUs))

                # is_best = mIoU > self.best_mIoU
                is_best = valid_acc > self.best_acc
                msg1 = "train loss: {:.3f} - train acc: {:.3f} - zeros acc: {:.3f} - ones acc: {:.3f}\n"
                # msg2 = "- val loss: {:.3f} - val acc: {:.3f} - val_mIoU: {:.3f}"
                msg2 = "- val loss: {:.3f} - val acc: {:.3f} - val_zeros acc: {:.3f} - val_ones acc: {:.3f}"
                if is_best:
                    self.counter = 0
                    msg2 += " [*]"
                msg = msg1 + msg2
                # print(msg.format(train_loss, train_acc, valid_loss, valid_acc, mIoU))
                print(msg.format(train_loss, train_acc, zeros_acc, ones_acc,
                                 valid_loss, valid_acc, val_zeros_acc, val_ones_acc))

                # check for improvement
                if not is_best:
                    self.counter += 1
                if self.counter > self.train_patience:
                    print("[!] No improvement in a while, stopping training.")
                    return
                # self.best_mIoU = max(mIoU, self.best_mIoU)
                self.best_acc = max(valid_acc, self.best_acc)
                self.save_checkpoint(
                    {'epoch': epoch + 1,
                     'model_state': self.model.state_dict(),
                     'optim_state': self.optimizer.state_dict(),
                     # 'best_valid_mIoU': self.best_mIoU,
                     'best_acc': self.best_acc
                     }, is_best
                )
        else:
            for epoch in range(self.start_epoch, self.epochs):

                print(
                    '\nEpoch: {}/{} - LR: {:.6f}'.format(
                        epoch + 1, self.epochs, self.lr)
                )

                # train for 1 epoch
                # train_loss, train_acc = self.train_one_epoch(epoch)
                train_loss, train_acc, rand_acc, maj_acc = self.train_one_epoch(epoch)

                # evaluate on validation set
                # valid_loss, valid_acc = self.validate(epoch)
                valid_loss, valid_acc, val_rand_acc, val_maj_acc = self.validate(epoch)

                # mIoU = (np.mean(valid_IoUs))

                # is_best = mIoU > self.best_mIoU
                is_best = valid_acc > self.best_acc
                msg1 = "train loss: {:.3f} - train acc: {:.3f} - rand acc: {:.3f} - maj acc: {:.3f}\n"
                # msg2 = "- val loss: {:.3f} - val acc: {:.3f} - val_mIoU: {:.3f}"
                msg2 = "- val loss: {:.3f} - val acc: {:.3f} - val rand acc: {:.3f} - val maj acc: {:.3f}"
                if is_best:
                    self.counter = 0
                    msg2 += " [*]"
                msg = msg1 + msg2
                # print(msg.format(train_loss, train_acc, valid_loss, valid_acc, mIoU))
                print(msg.format(train_loss, train_acc, rand_acc, maj_acc,
                                 valid_loss, valid_acc, val_rand_acc, val_maj_acc))

                # check for improvement
                if not is_best:
                    self.counter += 1
                if self.counter > self.train_patience:
                    print("[!] No improvement in a while, stopping training.")
                    return
                # self.best_mIoU = max(mIoU, self.best_mIoU)
                self.best_acc = max(valid_acc, self.best_acc)
                self.save_checkpoint(
                    {'epoch': epoch + 1,
                     'model_state': self.model.state_dict(),
                     'optim_state': self.optimizer.state_dict(),
                     # 'best_valid_mIoU': self.best_mIoU,
                     'best_acc': self.best_acc
                     }, is_best
                )

    def train_one_epoch(self, epoch):
        """
        Train the model for 1 epoch of the training set.

        An epoch corresponds to one full pass through the entire
        training set in successive mini-batches.

        This is used by train() and should not be called manually.
        """
        batch_time = AverageMeter()
        losses = AverageMeter()
        accs = AverageMeter()
        all_zeros = AverageMeter()
        all_ones = AverageMeter()
        all_rand = AverageMeter()
        all_majority = AverageMeter()

        tic = time.time()
        with tqdm(total=self.num_train) as pbar:
            for i, (x, y) in enumerate(self.train_loader):
                if self.config.binary:
                    x, y = Variable(x).float(), Variable(y).float()
                else:
                    x, y = Variable(x).float(), Variable(y).long()
                if self.use_gpu:
                    x, y = x.cuda(), y.cuda()

                self.batch_size = x.shape[0]
                x = x.view(self.batch_size, self.pc_size, self.input_dim)

                # Do the masking of indices to create the semi-supervised learning problem
                if self.config.semi:
                    x, mask_indices = self.mask_tensor(x, self.mask_rate)

                out = self.model(x)

                # TODO: Instead of squeeze, change view to handle tensors of batch_size != 1
                # To calculate loss, we retrieve everything from 4th column to end
                if self.config.semi:
                    pred = out.squeeze()[mask_indices]
                    labels = y.squeeze()[mask_indices]
                else:
                    pred = out.squeeze()
                    labels = y.squeeze()

                if self.config.binary:
                    loss = self.loss(pred, labels)

                    # compute accuracy
                    predicted = torch.max(pred, 1)[1]
                    true = torch.max(labels, 1)[1]
                    correct = (predicted == true).float()
                    acc = 100 * (correct.sum() / labels.shape[0])

                    predicted = torch.zeros(labels.shape[0], dtype=torch.long)
                    if self.use_gpu:
                        predicted = predicted.cuda()
                    correct = (predicted == true).float()
                    acc_zeros = 100 * (correct.sum() / labels.shape[0])

                    predicted = torch.ones(labels.shape[0], dtype=torch.long)
                    if self.use_gpu:
                        predicted = predicted.cuda()
                    correct = (predicted == true).float()
                    acc_ones = 100 * (correct.sum() / labels.shape[0])

                    all_zeros.update(acc_zeros.item(), labels.size()[0])
                    all_ones.update(acc_ones.item(), labels.size()[0])
                else:
                    labels = torch.max(labels, 1)[1]
                    loss = self.loss(pred, labels)

                    # compute accuracy
                    predicted = torch.max(pred, 1)[1]
                    true = labels
                    correct = (predicted == true).float()
                    acc = 100 * (correct.sum() / labels.shape[0])

                    # For the 1-of-8 problem, we use a random tensor as a baseline
                    # as well as a majority class tensor
                    predicted = torch.zeros(labels.shape[0], dtype=torch.long).random_(0, 8)
                    if self.use_gpu:
                        predicted = predicted.cuda()
                    correct = (predicted == true).float()
                    acc_rand = 100 * (correct.sum() / labels.shape[0])

                    predicted = torch.zeros(labels.shape[0], dtype=torch.long)
                    if self.use_gpu:
                        predicted = predicted.cuda()
                    correct = (predicted == true).float()
                    acc_maj = 100 * (correct.sum() / labels.shape[0])

                    all_rand.update(acc_rand.item(), labels.size()[0])
                    all_majority.update(acc_maj.item(), labels.size()[0])

                # store
                losses.update(loss.item(), labels.size()[0])
                accs.update(acc.item(), labels.size()[0])

                # compute gradients and update SGD
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # measure elapsed time
                toc = time.time()
                batch_time.update(toc-tic)

                pbar.set_description(
                    (
                        "{:.1f}s - loss: {:.3f} - acc: {:.3f}".format(
                            (toc-tic), loss.item(), acc.item()
                        )
                    )
                )
                pbar.update(self.batch_size)

            if self.config.binary:
                return losses.avg, accs.avg, all_zeros.avg, all_ones.avg
            else:
                return losses.avg, accs.avg, all_rand.avg, all_majority.avg

    def validate(self, epoch):
        """
        Evaluate the model on the validation set.
        """
        losses = AverageMeter()
        accs = AverageMeter()
        all_zeros = AverageMeter()
        all_ones = AverageMeter()
        all_rand = AverageMeter()
        all_majority = AverageMeter()

        for i, (x, y) in enumerate(self.valid_loader):
            if self.config.binary:
                x, y = Variable(x).float(), Variable(y).float()
            else:
                x, y = Variable(x).float(), Variable(y).long()
            if self.use_gpu:
                x, y = x.cuda(), y.cuda()


            self.batch_size = x.shape[0]
            x = x.view(self.batch_size, self.pc_size, self.input_dim)

            if self.config.semi:
                # Do the masking of indices to create the semi-supervised learning problem
                x, mask_indices = self.mask_tensor(x, self.mask_rate)

            out = self.model(x)

            # TODO: Instead of squeeze, change view to handle tensors of batch_size != 1
            # To calculate loss, we retrieve everything from 4th column to end
            if self.config.semi:
                pred = out.squeeze()[mask_indices]
                labels = y.squeeze()[mask_indices]
            else:
                pred = out.squeeze()
                labels = y.squeeze()

            if self.config.binary:
                loss = self.loss(pred, labels)

                # compute accuracy
                predicted = torch.max(pred, 1)[1]
                true = torch.max(labels, 1)[1]
                correct = (predicted == true).float()
                acc = 100 * (correct.sum() / labels.shape[0])

                predicted = torch.zeros(labels.shape[0], dtype=torch.long)
                if self.use_gpu:
                    predicted = predicted.cuda()
                correct = (predicted == true).float()
                acc_zeros = 100 * (correct.sum() / labels.shape[0])

                predicted = torch.ones(labels.shape[0], dtype=torch.long)
                if self.use_gpu:
                    predicted = predicted.cuda()
                correct = (predicted == true).float()
                acc_ones = 100 * (correct.sum() / labels.shape[0])

                all_zeros.update(acc_zeros.item(), labels.size()[0])
                all_ones.update(acc_ones.item(), labels.size()[0])
            else:
                labels = torch.max(labels, 1)[1]
                loss = self.loss(pred, labels)

                # compute accuracy
                predicted = torch.max(pred, 1)[1]
                true = labels
                correct = (predicted == true).float()
                acc = 100 * (correct.sum() / labels.shape[0])

                # For the 1-of-8 problem, we use a random tensor as a baseline
                # as well as a majority class tensor
                predicted = torch.zeros(labels.shape[0], dtype=torch.long).random_(0, 8)
                if self.use_gpu:
                    predicted = predicted.cuda()
                correct = (predicted == true).float()
                acc_rand = 100 * (correct.sum() / labels.shape[0])

                predicted = torch.zeros(labels.shape[0], dtype=torch.long)
                if self.use_gpu:
                    predicted = predicted.cuda()
                correct = (predicted == true).float()
                acc_maj = 100 * (correct.sum() / labels.shape[0])

                all_rand.update(acc_rand.item(), labels.size()[0])
                all_majority.update(acc_maj.item(), labels.size()[0])

            # store
            losses.update(loss.item(), labels.size()[0])
            accs.update(acc.item(), labels.size()[0])

        if self.config.binary:
            return losses.avg, accs.avg, all_zeros.avg, all_ones.avg
        else:
            return losses.avg, accs.avg, all_rand.avg, all_majority.avg

    def test(self):
        total_acc = 0
        total_zeros = 0
        total_ones = 0
        total_rand = 0
        total_majority = 0
        total_num_points = 0

        # load the best checkpoint
        self.load_checkpoint(best=self.best)

        for i, (x, y) in enumerate(self.valid_loader):
            if self.config.binary:
                x, y = Variable(x).float(), Variable(y).float()
            else:
                x, y = Variable(x).float(), Variable(y).long()
            if self.use_gpu:
                x, y = x.cuda(), y.cuda()

            # initialize location vector and hidden state
            self.batch_size = x.shape[0]
            x = x.view(self.batch_size, self.pc_size, self.input_dim)

            # Do the masking of indices to create the semi-supervised learning problem
            if self.config.semi:
                x, mask_indices = self.mask_tensor(x, self.mask_rate)

            out = self.model(x)

            # TODO: Instead of squeeze, change view to handle tensors of batch_size != 1
            # To calculate loss, we retrieve everything from 4th column to end
            if self.config.semi:
                pred = out.squeeze()[mask_indices]
                labels = y.squeeze()[mask_indices]
                total_num_points += labels.shape[0]
            else:
                pred = out.squeeze()
                labels = y.squeeze()
                total_num_points += labels.shape[0]

            # compute accuracy
            predicted = torch.max(pred, 1)[1]
            true = torch.max(labels, 1)[1]
            correct = (predicted == true).float()
            total_acc += correct.sum()


            if self.config.binary:
                predicted = torch.zeros(labels.shape[0], dtype=torch.long)
                if self.use_gpu:
                    predicted = predicted.cuda()
                correct = (predicted == true).float()
                total_zeros += correct.sum()

                predicted = torch.ones(labels.shape[0], dtype=torch.long)
                if self.use_gpu:
                    predicted = predicted.cuda()
                correct = (predicted == true).float()
                total_ones += correct.sum()
            else:
                # For the 1-of-8 problem, we use a random tensor as a baseline
                # as well as a majority class baseline
                predicted = torch.zeros(labels.shape[0], dtype=torch.long).random_(0, 8)
                if self.use_gpu:
                    predicted = predicted.cuda()
                correct = (predicted == true).float()
                total_rand += correct.sum()

                predicted = torch.zeros(labels.shape[0], dtype=torch.long)
                if self.use_gpu:
                    predicted = predicted.cuda()
                correct = (predicted == true).float()
                total_majority += correct.sum()


            print("Done with %.3f%%" % ((i + 1) / self.num_valid * 100.))

        print()

        if self.config.binary:
            msg = "Final Accuracy: {:.3f} - Background Baseline: {:.3f} - Foreground Baseline: {:.3f}\n"
            print(msg.format(total_acc / total_num_points,
                             total_zeros / total_num_points,
                             total_ones / total_num_points))
        else:
            msg = "Final Accuracy: {:.3f} - Random Baseline: {:.3f} - Majority Class Baseline: {:.3f}\n"
            print(msg.format(total_acc / total_num_points,
                             total_rand / total_num_points,
                             total_majority / total_num_points))

    def save_checkpoint(self, state, is_best):
        """
        Save a copy of the model so that it can be loaded at a future
        date. This function is used when the model is being evaluated
        on the test data.

        If this model has reached the best validation accuracy thus
        far, a seperate file with the suffix `best` is created.
        """
        # print("[*] Saving model to {}".format(self.ckpt_dir))

        filename = self.model_name + '_ckpt.pth.tar'
        ckpt_path = os.path.join(self.ckpt_dir, filename)
        torch.save(state, ckpt_path)

        if is_best:
            filename = self.model_name + '_model_best.pth.tar'
            shutil.copyfile(
                ckpt_path, os.path.join(self.ckpt_dir, filename)
            )

    def load_checkpoint(self, best=False):
        """
        Load the best copy of a model. This is useful for 2 cases:

        - Resuming training with the most recent model checkpoint.
        - Loading the best validation model to evaluate on the test data.

        Params
        ------
        - best: if set to True, loads the best model. Use this if you want
          to evaluate your model on the test data. Else, set to False in
          which case the most recent version of the checkpoint is used.
        """
        print("[*] Loading model from {}".format(self.ckpt_dir))

        filename = self.model_name + '_ckpt.pth.tar'
        if best:
            filename = self.model_name + '_model_best.pth.tar'
        ckpt_path = os.path.join(self.ckpt_dir, filename)
        ckpt = torch.load(ckpt_path)

        # load variables from checkpoint
        self.start_epoch = ckpt['epoch']
        # self.best_mIoU = ckpt['best_valid_mIoU']
        self.model.load_state_dict(ckpt['model_state'])
        self.optimizer.load_state_dict(ckpt['optim_state'])

        print("Successfully loaded model...")

        if best:
            print(
                "[*] Loaded {} checkpoint @ epoch {} "
                "with best valid acc of {:.3f}".format(
                    filename, ckpt['epoch'], ckpt['best_acc'])
            )
        else:
            print(
                "[*] Loaded {} checkpoint @ epoch {}".format(
                    filename, ckpt['epoch'])
            )
