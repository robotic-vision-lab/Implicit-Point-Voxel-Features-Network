import models.network as model
import models.data.voxelized_data_shapenet as voxelized_data
from models.generation import Generator
import torch
import configs.config_loader as cfg_loader
import os
from os.path import join
import trimesh
import numpy as np
from tqdm import tqdm

cfg = cfg_loader.get_config()

device = torch.device("cuda")
net = model.IPVNet(cfg)

print(cfg)
print(net)

dataset = voxelized_data.VoxelizedDataset('test',
                                          res=cfg.input_res,
                                          pointcloud_samples=cfg.num_points,
                                          data_path=cfg.data_dir,
                                          split_file=cfg.split_file,
                                          batch_size=1,
                                          num_sample_points=cfg.num_sample_points_generation,
                                          num_workers=1,
                                          sample_distribution=cfg.sample_ratio,
                                          sample_sigmas=cfg.sample_std_dev)

gen = Generator(net, cfg.exp_name, results_path=cfg.results_dir,
                checkpoint=cfg.checkpoint, device=device)

out_path = join(cfg.results_dir, cfg.exp_name, 'evaluation')


def gen_iterator(out_path, dataset, gen):

    if not os.path.exists(out_path):
        os.makedirs(out_path)

    # can be run on multiple machines: dataset is shuffled and already generated objects are skipped.
    loader = dataset.get_loader(shuffle=False)

    for i, data in enumerate(loader):
        path = os.path.normpath(data['path'][0])
        export_path = join(
            out_path, 'generation/{}/{}/'.format(path.split(os.sep)[-2], path.split(os.sep)[-1]))

        if not os.path.exists(export_path):
            os.makedirs(export_path)

        for num_steps in [7]:
            off_path = f'{export_path}dense_point_cloud_{cfg.num_points}points_{num_steps}steps.off'
            if os.path.exists(off_path):
                print(f'Path exists - skip! {off_path}')
                continue

            point_cloud, duration = gen.generate_point_cloud(data, num_steps)

            np.savez(join(export_path, 'dense_point_cloud_{}points_{}steps'.format(
                cfg.num_points, num_steps)), point_cloud=point_cloud, duration=duration)
            print('iter', i, 'total', len(loader), 'file', os.path.basename(
                export_path[:-1]), 'num_steps', num_steps, 'duration', duration)
            trimesh.Trimesh(vertices=point_cloud, faces=[]).export(off_path)


gen_iterator(out_path, dataset, gen)
