from __future__ import division
import torch
import torch.optim as optim
from torch.nn import functional as F
import os
from os.path import join as join
from torch.utils.tensorboard import SummaryWriter
from glob import glob
import numpy as np
import time


from torch.nn import DataParallel as DP


class Trainer(object):

    def __init__(self, model, device, train_dataset, val_dataset, exp_name, optimizer='Adam', lr=1e-4, threshold=0.1, cfg=None):
        self.model = DP(model).to(device)
        self.device = device
        if optimizer == 'Adam':
            self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        if optimizer == 'Adadelta':
            self.optimizer = optim.Adadelta(self.model.parameters())
        if optimizer == 'RMSprop':
            self.optimizer = optim.RMSprop(
                self.model.parameters(), momentum=0.9)

        self.cfg = cfg
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.exp_path = join(cfg.results_dir, exp_name)
        self.checkpoint_path = join(self.exp_path, 'checkpoints')
        if not os.path.exists(self.checkpoint_path):
            print(self.checkpoint_path)
            os.makedirs(self.checkpoint_path)
        self.writer = SummaryWriter(join(self.exp_path, 'summary'))
        self.writer.add_text('args', str(cfg), 0)
        self.writer.add_text('args', str(model), 0)

        self.val_min = None
        self.max_dist = threshold

        if cfg.loss == 'l1':
            self.loss_f = torch.nn.L1Loss(reduction='none')
        else:
            self.loss_f = torch.nn.MSELoss(reduction='none')

    def switch_grad(self, value):
        for p in self.model.parameters():
            p.requires_grad = value

    def train_step(self, batch):
        self.model.train()
        self.optimizer.zero_grad()
        loss = self.compute_loss(batch)
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def compute_loss(self, batch):
        device = self.device
        p = batch.get('grid_coords').to(device)
        df_gt = batch.get('df').to(device)              # (B, S)
        inputs = batch.get('inputs').to(device)         # (B, R, R, R)
        points = batch.get('point_cloud').to(device)    # (B, N, 3)
        # print(torch.cuda.memory_allocated(), torch.cuda.memory_reserved())
        df_pred = self.model(p, inputs, points)           # (B, S)
        # print(torch.cuda.memory_allocated(), torch.cuda.memory_reserved())
        # summed over all #num_samples_training -> out = (B,1) and mean over batch -> out = (1)
        loss = self.loss_f(torch.clamp(df_pred, max=self.max_dist), torch.clamp(
            df_gt, max=self.max_dist)).sum(-1).mean()
        # print(torch.cuda.memory_allocated(), torch.cuda.memory_reserved())
        return loss

    def train_model(self, epochs):
        loss = 0
        train_data_loader = self.train_dataset.get_loader()
        start, training_time = self.load_checkpoint()
        iteration_start_time = time.time()

        for epoch in range(start, epochs):
            sum_loss = 0
            t_val_loss, val_count = 0, 0

            for _iter, batch in enumerate(train_data_loader):

                iteration_duration = time.time() - iteration_start_time

                # optimize model
                loss = self.train_step(batch)

                # timming
                iter_net_time = time.time()
                eta = ((iter_net_time - iteration_start_time) / (_iter + 1)) * len(train_data_loader) - \
                    (iter_net_time - iteration_start_time)  # remaining sec(s) for this epoch

                print(
                    f"Epoch: {epoch:3d} Iter {_iter:3d} Mean {self.cfg.loss} Loss: {loss:5.5f} Iter dur: {iteration_duration:5.5f} ETA: {eta/3600:5.5f}")

                sum_loss += loss

                # save model
                # save model every X min and at the end of iteration
                if iteration_duration > 60 * 60 or _iter == len(train_data_loader)-1:
                    training_time += iteration_duration
                    iteration_start_time = time.time()

                    self.save_checkpoint(epoch, training_time)
                    with torch.no_grad():
                        val_loss = self.compute_val_loss()

                    t_val_loss += val_loss
                    val_count += 1

                    if self.val_min is None:
                        self.val_min = val_loss

                    if val_loss < self.val_min:
                        self.val_min = val_loss
                        for path in glob(join(self.exp_path, 'val_min_*')):
                            os.remove(path)
                        np.save(join(self.exp_path, 'val_min_{}_{}h:{}m:{}s_{}'.format(
                            val_loss, *[*convertSecs(training_time), training_time])), [epoch, val_loss])

                    print(
                        f'Epoch: {epoch:3d} iter: {_iter:3d} Val loss: {val_loss:1.5f}')
                    if _iter == len(train_data_loader)-1:
                        print(
                            f'Epoch: {epoch:3d} Iter: {_iter:3d} Val loss batch avg: {val_loss:1.5f}')
                        self.writer.add_scalar(
                            'val loss batch avg', t_val_loss/val_count, epoch)

            print(
                f'Epoch: {epoch:3d} Training loss batch avg: {sum_loss / len(train_data_loader): 1.5f}')
            self.writer.add_scalar(
                'training loss batch avg', sum_loss / len(train_data_loader), epoch)

        self.writer.close()

    def validate_model(self, start_hr=None, end_hr=None):
        self.model.eval()

        val_data_loader = self.val_dataset.get_loader()
        num_batch = len(val_data_loader)

        checkpoints = self.get_checkpoints(start_hr, end_hr)
        min_val_loss = 10000

        for count, ch in enumerate(checkpoints):
            sum_val_loss = 0
            epoch, training_time = self.load_checkpoint(ch)
            for _iter, data in enumerate(val_data_loader):
                sum_val_loss += self.compute_loss(data).item()

            avg_val_loss = sum_val_loss / num_batch
            if avg_val_loss < min_val_loss:
                min_val_loss = avg_val_loss
                min_epoch = epoch
                min_train_time = training_time

            print(
                f'{count}/{len(checkpoints)} Checkpoint: {ch} Mean Val Loss {avg_val_loss:5.5f}')
            self.writer.add_scalar('Mean Val Loss', avg_val_loss, epoch)

        self.writer.add_text('Min Val Loss', str(min_val_loss), 1)
        self.writer.add_text('Min Val Epoch', str(min_epoch), 1)
        self.writer.add_text('Min Val Training Time', str(min_train_time), 1)
        print(
            f'Min val loss: {min_val_loss} Epoch: {min_epoch} Time: {min_train_time}')
        self.writer.close()

    def save_checkpoint(self, epoch, training_time):
        path = join(self.checkpoint_path, 'checkpoint_{}h:{}m:{}s_{}.tar'.format(
            *[*convertSecs(training_time), training_time]))
        if not os.path.exists(path):
            torch.save({  # 'state': torch.cuda.get_rng_state_all(),
                'training_time': training_time, 'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict()}, path)

    def get_checkpoints(self, start_hr=None, end_hr=None):
        '''
        Returns list of all/range(start_hr, end_hr) checkpoints (if exists) in checkpoint path.

        '''

        checkpoints = glob(self.checkpoint_path+'/*')
        if len(checkpoints) == 0:    # No checkpoint found
            return checkpoints

        checkpoints = [os.path.splitext(os.path.basename(path))[0].split(
            '_')[-1] for path in checkpoints]    # checkpoints = [0, 360, 720, ...]
        checkpoints = np.array(checkpoints, dtype=float)
        checkpoints = np.sort(checkpoints)

        if start_hr == None:
            return checkpoints    # all checkpoints

        checkpoints = checkpoints[checkpoints > start_hr]
        if end_hr == None:
            return checkpoints    # all checkpoints > chosen start hr

        checkpoints = checkpoints[checkpoints < end_hr]
        return checkpoints    # all checkpoints in chosen (start ~ end) hr

    def load_checkpoint(self, checkpoint=None):
        '''
        Loads chosen/latest checkpoint.
        '''

        if checkpoint == None:    # No checkpoint is chosen
            checkpoints = self.get_checkpoints()    # get all checkpoints
            if len(checkpoints) == 0:    # no chekcpoint exists. Start from scratch
                print('No checkpoints found at {}'.format(self.checkpoint_path))
                return 0, 0    # start epoch - 0, training time - 0
        else:    # checkpoint chosen, making list
            checkpoints = [checkpoint]

        path = join(self.checkpoint_path, 'checkpoint_{}h:{}m:{}s_{}.tar'.format(
            *[*convertSecs(checkpoints[-1]), checkpoints[-1]]))

        print('Loading checkpoint from: {}'.format(path))
        checkpoint = torch.load(path)
        self.model.load_state_dict(
            checkpoint['model_state_dict'], strict=False)
        # print(checkpoint['optimizer_state_dict']['param_groups'], self.optimizer.state_dict()['param_groups'])

        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        training_time = checkpoint['training_time']
        # torch.cuda.set_rng_state_all(checkpoint['state']) # batch order is restored. unfortunately doesn't work like that.
        return epoch, training_time
        # return 0, 0.0

    def compute_val_loss(self):
        self.model.eval()

        sum_val_loss = 0
        num_batches = min(25, int(len(self.val_dataset) *
                          0.25) // self.val_dataset.batch_size)

        for _ in range(num_batches):
            try:
                val_batch = self.val_data_iterator.next()
            except:
                self.val_data_iterator = self.val_dataset.get_loader().__iter__()
                val_batch = self.val_data_iterator.next()

            with torch.no_grad():
                sum_val_loss += self.compute_loss(val_batch).item()

        return sum_val_loss / num_batches


def convertMillis(millis):
    seconds = int((millis / 1000) % 60)
    minutes = int((millis / (1000 * 60)) % 60)
    hours = int((millis / (1000 * 60 * 60)))
    return hours, minutes, seconds


def convertSecs(sec):
    seconds = int(sec % 60)
    minutes = int((sec / 60) % 60)
    hours = int((sec / (60 * 60)))
    return hours, minutes, seconds
