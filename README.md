# Lost in Latent Space

This repository contains the official implementation of the paper [Lost in Latent Space: An Empirical Study of Latent Diffusion Models for Physics Emulation](https://arxiv.org/abs/TODO) by [François Rozet](https://github.com/francois-rozet), [Ruben Ohana](https://github.com/rubenohana), [Michael McCabe](https://github.com/mikemccabe210), [Gilles Louppe](https://github.com/glouppe), [François Lanusse](https://github.com/EiffL), and [Shirley Ho](https://github.com/shirleysurelyho).

#### Abstract

The steep computational cost of diffusion models at inference hinders their use as fast physics emulators. In the context of image and video generation, this computational drawback has been addressed by generating in the latent space of an autoencoder instead of the pixel space. In this work, we investigate whether a similar strategy can be effectively applied to the emulation of dynamical systems and at what cost. We find that the accuracy of latent-space emulation is surprisingly robust to a wide range of compression rates (up to 1000x). We also show that diffusion-based emulators are consistently more accurate than non-generative counterparts and compensate for uncertainty in their predictions with greater diversity. Finally, we cover practical design choices, spanning from architectures to optimizers, that we found critical to train latent-space emulators.

<p align="center"><img src="assets/emulation.svg" width="95%"></p>

## Code

The majority of the code is written in [Python](https://www.python.org). Neural networks are implemented and trained using the [PyTorch](https://github.com/pytorch/pytorch) automatic differentiation framework. To run the experiments, it is necessary to have access to a [Slurm](https://slurm.schedmd.com) cluster, to login to a [Weights & Biases](https://wandb.ai) account and to install the [lola](lola) module as a package.

First, create a new Python environment, for example with [venv](https://docs.python.org/3/library/venv.html).

```
python -m venv ~/.venvs/lola
source ~/.venvs/lola/bin/activate
```

Then, install the [lola](lola) module as an [editable](https://pip.pypa.io/en/latest/topics/local-project-installs) package with its dependencies.

```
pip install --editable .[all] --extra-index-url https://download.pytorch.org/whl/cu121
```

Optionally, we provide [pre-commit hooks](pre-commit.yaml) to automatically detect code issues.

```
pre-commit install --config pre-commit.yaml
```

### Organization

The [lola](lola) directory contains the implementations of the [neural networks](lola/nn), the [autoencoders](lola/autoencoders.py), the [diffusion models](lola/diffusion.py), the [emulation routines](lola/emulation.py), and others.

The [experiments](experiments) directory contains the training scripts, the evaluation scripts and their [configurations](experiments/configs). The [euler](experiments/euler) and [rayleigh_benard](experiments/rayleigh_benard) directories contain the notebooks that produced the figures of the paper.

### Data

We rely on a [Ceph File System](https://docs.ceph.com/en/latest/cephfs) partition to store the data. If your cluster uses a different file system, we recommend to create a symbolic link in your home folder.

```
ln -s /mnt/filesystem/users/you ~/ceph
```

The datasets (Euler and Rayleigh-Bénard) are downloaded from [The Well](https://github.com/PolymathicAI/the_well).

```
the-well-download --base-path ~/ceph/the_well --dataset euler_multi_quadrants_openBC
the-well-download --base-path ~/ceph/the_well --dataset euler_multi_quadrants_periodicBC
the-well-download --base-path ~/ceph/the_well --dataset rayleigh_benard
```

> This could take a while!

### Experiments

We start with training the autoencoders. For clarity, we provide the commands for a single compression rate. To replicate the other experiments, modify the number of latent channels.

```
python train_ae.py dataset=euler_all optim.learning_rate=1e-5 ae.lat_channels=64
python train_ae.py dataset=rayleigh_benard optim.learning_rate=1e-5 ae.lat_channels=64
```

Once the above jobs are completed (1-2 days), we encode the entire dataset with each trained autoencoder and cache the resulting latent trajectories permanently on disk. For instance, for the autoencoder run named `di2j3rpb_rayleigh_benard_dcae_f32c64_large`,

```
python cache_latents.py dataset=rayleigh_benard split=train repeat=4 run=~/ceph/lola/runs/ae/di2j3rpb_rayleigh_benard_dcae_f32c64_large
python cache_latents.py dataset=rayleigh_benard split=valid run=~/ceph/lola/runs/ae/di2j3rpb_rayleigh_benard_dcae_f32c64_large
```

The stored latent trajectories are then used to train latent-space emulators (deterministic and diffusion-based), without needing to load and encode high-dimensional samples on the fly.

```
python train_surrogate.py dataset=rayleigh_benard ae_run=~/ceph/lola/runs/ae/di2j3rpb_rayleigh_benard_dcae_f32c64_large  # neural solver
python train_diffusion.py dataset=rayleigh_benard ae_run=~/ceph/lola/runs/ae/di2j3rpb_rayleigh_benard_dcae_f32c64_large  # diffusion model
```

We also train pixel-space deterministic emulators, which require more compute resources.

```
python train_surrogate.py dataset=euler_all surrogate=vit_pixel compute.nodes=2
python train_surrogate.py dataset=rayleigh_benard surrogate=vit_pixel compute.nodes=2
```

Finally, we evaluate each trained emulator on the test set.

```
python eval.py start=16 seed=0 run=~/ceph/lola/runs/sm/lrg1qgi2_rayleigh_benard_f32c64_vit_large  # neural solver
python eval.py start=16 seed=0 run=~/ceph/lola/runs/dm/0fqjt3js_rayleigh_benard_f32c64_vit_large  # diffusion model
```

> Each `train_*.py` script schedules a Slurm job to train a model, log the training statistics with `wandb`, and store the weights in the `~/ceph/lola/runs` directory. You will likely have to adapt the requested resources, either in the [config files](experiments/configs) or in the command line.
