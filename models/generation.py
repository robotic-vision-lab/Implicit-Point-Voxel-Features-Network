import torch
import os
from os.path import join
from glob import glob
import numpy as np
from torch.nn import functional as F
import time
from torch.nn import DataParallel as DP
from torch import distributions as dist


class Generator(object):
    def __init__(self, model, exp_name, threshold=0.1, results_path=None, checkpoint=None, device=torch.device("cuda")):
        self.model = DP(model).to(device)
        self.model.eval()
        self.device = device
        self.checkpoint_path = join(results_path, exp_name, 'checkpoints')
        # e, t = self.load_checkpoint(checkpoint)
        # print(e, t)
        self.threshold = threshold

    def generate_point_cloud(self, data, num_steps=10, num_points=1000000, filter_val=0.009):

        start = time.time()
        sample_num = 100000

        inputs = data['inputs'].to(self.device)
        point_cloud = data['point_cloud'].to(self.device)

        points = torch.index_select(
            data['point_cloud'], -1, torch.LongTensor([2, 1, 0])).to(self.device) * 3
        points = points[:, torch.randint(0, points.shape[1], [sample_num]), :]
        num_points = points.shape[1]*5
        
        for param in self.model.parameters():
            param.requires_grad = False

        samples_cpu = np.zeros((0, 3))
        samples = points + \
            torch.rand(1, sample_num, 3).float().to(self.device) * 2.0 - 1.0

        samples.requires_grad = True
        encoding = self.model.module.encoder(inputs, point_cloud)

        i = 0
        while len(samples_cpu) < num_points:
            print('iteration', i, 'samples_cpu', samples_cpu.shape)

            for j in range(num_steps):
                df_pred = torch.clamp(self.model.module.decoder(
                    samples, encoding), max=self.threshold)
                print('refinement', j, df_pred.mean().item())

                df_pred.sum().backward()

                gradient = samples.grad.detach()
                samples = samples.detach()
                df_pred = df_pred.detach()
                inputs = inputs.detach()
                
                samples = samples - \
                    F.normalize(gradient, dim=2) * df_pred.reshape(-1, 1)
                samples = samples.detach()
                samples.requires_grad = True

            if i > 0:
                samples_cpu = np.vstack(
                    (samples_cpu, samples[df_pred < filter_val].detach().cpu().numpy()))

            else:
                samples = samples[df_pred < 0.03]
                samples = samples.unsqueeze(0)

                indices = torch.randint(samples.shape[1], (1, sample_num))
                samples = samples[[[0, ] * sample_num], indices]
                samples += (self.threshold / 3) * \
                    torch.randn(samples.shape).to(self.device)
                samples = samples.detach()
                samples.requires_grad = True

            i += 1

        duration = time.time() - start

        return samples_cpu, duration

    def load_checkpoint(self, checkpoint):
        checkpoints = glob(self.checkpoint_path + '/*')
        if checkpoint is None:
            if len(checkpoints) == 0:
                print('No checkpoints found at {}'.format(self.checkpoint_path))
                return 0, 0

            checkpoints = [os.path.splitext(os.path.basename(path))[
                0].split('_')[-1] for path in checkpoints]
            checkpoints = np.array(checkpoints, dtype=float)
            checkpoints = np.sort(checkpoints)
            path = join(self.checkpoint_path, 'checkpoint_{}h:{}m:{}s_{}.tar'.format(
                *[*convertSecs(checkpoints[-1]), checkpoints[-1]]))
        else:
            path = join(self.checkpoint_path, '{}.tar'.format(checkpoint))
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        epoch = checkpoint['epoch']
        training_time = checkpoint['training_time']
        print('Loaded checkpoint from: {} at epoch: {}'.format(path, epoch))
        return epoch, training_time


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
