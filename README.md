# MPP x Latent Diffusion Models

## Code

To run the experiments, it is necessary to have access to a [Slurm](https://slurm.schedmd.com) cluster, to login to a [Weights & Biases](https://wandb.ai) account and to install the [lpdm](lpdm) module as a package. First, create a new Python environment, for example with [venv](https://docs.python.org/3/library/venv.html).

```
python -m venv ~/.venvs/lpdm
source ~/.venvs/lpdm/bin/activate
```

Then, install the [lpdm](lpdm) module as an [editable](https://pip.pypa.io/en/latest/topics/local-project-installs) package with its dependencies.

```
pip install --editable .[all] --extra-index-url https://download.pytorch.org/whl/cu121
```

Optionally, we provide [pre-commit hooks](pre-commit.yml) to automatically detect code issues.

```
pre-commit install --config pre-commit.yaml
```

Before running the [experiments](experiments/), create a symlink to the Well datasets.

```
ln -s /mnt/ceph/users/polymathic/the_well ~/ceph/the_well
```

## Experiments

```
cd experiments
```

### Auto-encoders

```
python train_ae.py dataset=euler_all ae.lat_channels=32
python train_ae.py dataset=euler_all ae.lat_channels=64
python train_ae.py dataset=euler_all ae.lat_channels=128
python train_ae.py dataset=rayleigh_benard ae.lat_channels=32
python train_ae.py dataset=rayleigh_benard ae.lat_channels=64
python train_ae.py dataset=rayleigh_benard ae.lat_channels=128
```

```
python cache_latents.py run=??? dataset=rayleigh_benard split=train repeat=4
python cache_latents.py run=??? dataset=rayleigh_benard dataset.augment=["log_scalars"] split=valid
```

### Pixel-space DMs

```
python train_dm.py dataset=euler_all
python train_dm.py dataset=rayleigh_benard
```

## Latent-space DMs

```
python train_ldm.py ae_from=??? dataset=rayleigh_benard
```
