import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import distributions as dist


def maxpool(x, dim=-1, keepdim=False):
    out, _ = x.max(dim=dim, keepdim=keepdim)
    return out


class PointEncoder(nn.Module):
    ''' PointNet-based encoder network. Based on: https://github.com/autonomousvision/occupancy_networks
    Args:
        c_dim (int): dimension of latent code c
        dim (int): input points dimension
        hidden_dim (int): hidden dimension of the network
    '''

    def __init__(self, L, c_dim=3, h_dim=256, l_dim=512):
        super().__init__()
        self.l_dim = l_dim
        self.num_layers = L

        self.fc = nn.ModuleDict()
        self.fc[f'fc_in_0'] = nn.Linear(c_dim, h_dim)

        for l in range(L):
            self.fc[f'fc_{l}'] = nn.Linear(2*h_dim, h_dim)
            if l == L-1:    # last layer
                self.fc[f'fc_out'] = nn.Linear(2*h_dim, l_dim)

        self.actvn = nn.ReLU()
        self.pool = maxpool

    def forward(self, p):
        features = []
        net = self.actvn(self.fc['fc_in_0'](p))

        for l in range(self.num_layers):
            features.append(self.pool(net, dim=1, keepdim=False))
            pooled = features[l].unsqueeze(1).expand(net.size())
            net = torch.cat([net, pooled], dim=2)
            if l == self.num_layers-1:
                net = self.actvn(self.fc[f'fc_out'](net))
                features.append(self.pool(net, dim=1, keepdim=False))

            else:
                net = self.actvn(self.fc[f'fc_{l}'](net))

        return features


class IPVNet(nn.Module):

    def __init__(self, opt):
        super(IPVNet, self).__init__()
        self.layers = opt.conv3d_layers

        self.point_encoder = PointEncoder(L=len(self.layers)-1,
                                          h_dim=opt.p_h_dim,
                                          l_dim=opt.p_l_dim)

        self.conv = nn.ModuleDict()
        self.fc = nn.ModuleDict()
        self.bn = nn.ModuleList()

        for l in range(len(self.layers)-1):
            self.conv[f'conv_{l}'] = nn.Conv3d(self.layers[l],
                                               self.layers[l+1], 3,
                                               padding=1)
            if l > 0:
                self.conv[f'conv_{l}_{0}'] = nn.Conv3d(self.layers[l+1]+opt.p_h_dim,
                                                       self.layers[l+1], 3,
                                                       padding=1)
            self.bn.append(nn.BatchNorm3d(self.layers[l+1]))

        feature_size = sum(self.layers) * 7 + 3 + opt.p_l_dim

        self.fc['fc_0'] = nn.Conv1d(feature_size, opt.h_dim * 2, 1)
        self.fc['fc_1'] = nn.Conv1d(opt.h_dim*2, opt.h_dim, 1)
        self.fc['fc_2'] = nn.Conv1d(opt.h_dim, opt.h_dim, 1)
        self.fc['fc_out'] = nn.Conv1d(opt.h_dim, 1, 1)
        self.actvn = nn.ReLU()

        self.maxpool = nn.MaxPool3d(2)

        displacment = 0.0722
        displacments = []
        displacments.append([0, 0, 0])
        for x in range(3):
            for y in [-1, 1]:
                input = [0, 0, 0]
                input[x] = y * displacment
                displacments.append(input)

        self.displacments = torch.Tensor(displacments).cuda()

    def encoder(self, x, points):
        f_p = self.point_encoder(points)
        features = []
        x = x.unsqueeze(1)
        features.append(x)

        for l in range(len(self.layers)-1):
            if l == 0:
                net = self.actvn(self.conv[f'conv_{l}'](x))
            else:
                net = self.actvn(self.conv[f'conv_{l}'](net))
            if l > 0:
                point_feature = f_p[l-1][:, :, None, None, None]
                target = net.shape[-1]
                point_feature = point_feature.repeat(
                    1, 1, target, target, target)
                net = torch.cat((net, point_feature), dim=1)
                net = self.actvn(self.conv[f'conv_{l}_0'](net))
            net = self.bn[l](net)
            features.append(net)
            net = self.maxpool(net)    # res = res//2

        features.append(f_p[-1])

        return features

    def decoder(self, p, feat):
        self.displacments = self.displacments.to(p.device)
        p_features = p.transpose(1, -1)
        p = p.unsqueeze(1).unsqueeze(1)
        p = torch.cat([p + d for d in self.displacments], dim=2)

        # feature extraction
        features = []
        f_p = feat[-1]
        feat = feat[:-1]

        for f in feat:
            features.append(F.grid_sample(
                f, p, padding_mode='border', align_corners=True))

        # here every channel corresponds to one feature.
        # (B, features, 1,7,sample_num)
        features = torch.cat((features), dim=1)
        shape = features.shape
        features = torch.reshape(features,
                                 (shape[0], shape[1] * shape[3], shape[4]))  # (B, featues_per_sample, samples_num)
        # (B, featue_size, samples_num)
        features = torch.cat((features, p_features), dim=1)
        features = torch.cat(
            (features, f_p.unsqueeze(-1).repeat(1, 1, features.size(-1))), dim=1)

        net = self.actvn(self.fc['fc_0'](features))
        net = self.actvn(self.fc['fc_1'](net))
        net = self.actvn(self.fc['fc_2'](net))

        net = self.actvn(self.fc['fc_out'](net))
        out = net.squeeze(1)

        return out

    def forward(self, p, x, points):
        out = self.decoder(p, self.encoder(x, points))
        return out
