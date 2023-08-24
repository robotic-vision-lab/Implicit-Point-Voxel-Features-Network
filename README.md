# Implicit-Point-Voxel-Features-Network

IPVNet: Learning Implicit Point-Voxel Features for Open-Surface 3D 
Reconstruction <br />
[Mohammad Samiul Arshad](https://samiarshad.github.io/), [William J. Beksi](https://ranger.uta.edu/~wjbeksi/)

Published in Journal of Visual Communication and Image Representation, 2023.

| 
[Paper]() |
[Vedio]() |
<!-- 
[Supplementaty]() -
[Project Website]() -
[Arxiv]() --->


<!-- ![Teaser](ndf-teaser.png) -->

#### Citation
If you find our project useful, please cite the following.


## Setup

Please clone the repository and navigate into it in your terminal.

The `environment.yml` file contains all necessary dependencies for the project.
```
conda env create -f environment.yml
conda activate IPVNet
```
## Data Processing

We have used the data processing scripts provided by the authors of [NDF](http://virtualhumans.mpi-inf.mpg.de/ndf/) [Chibane et. al. NeurIPS'19] to prepare the data for our experiments. Please follow their directions from [here](https://github.com/jchibane/ndf) to download and process the data.

## Experiment Preparation

To train IPVNet, create a configuration file in folder `configs/`(use `configs/shapenet_raw.txt` as reference) and generate a random test/training/validation split of the data using
```
python dataprocessing/create_split.py --config configs/EXP_NAME.txt
```
Replace `configs/EXP_NAME.txt` in the above commands with the desired configuration.

## Training and generation

Train IPVNet using
```
python train.py --config configs/EXP_NAME.txt
```

To generate results for instances of the test set, please use
```
python generate.py --config configs/EXP_NAME.txt
```
Again replace `configs/EXP_NAME.txt` in the above commands with the desired configuration.

## Contact

For questions and comments, please contact [Mohammad Samiul Arshad](https://samiarshad.github.io/) via mail.


## License

## Acknowledgement

Our code is based on [NDF](https://github.com/jchibane/ndf). We thank the authors for their excellent work!

