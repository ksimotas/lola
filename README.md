# Lost in Latent Space: An Empirical Study of Latent Diffusion Models for Physics Emulation

## Code

To run the experiments, it is necessary to have access to a [Slurm](https://slurm.schedmd.com) cluster, to login to a [Weights & Biases](https://wandb.ai) account and to install the [lola](lola) module as a package. First, create a new Python environment, for example with [venv](https://docs.python.org/3/library/venv.html).

```
python -m venv ~/.venvs/lola
source ~/.venvs/lola/bin/activate
```

Then, install the [lola](lola) module as an [editable](https://pip.pypa.io/en/latest/topics/local-project-installs) package with its dependencies.

```
pip install --editable .[all] --extra-index-url https://download.pytorch.org/whl/cu121
```

Optionally, we provide [pre-commit hooks](pre-commit.yml) to automatically detect code issues.

```
pre-commit install --config pre-commit.yaml
```

## Experiments

```
cd experiments
```

### Auto-encoders

```
python train_ae.py dataset=euler_all optim.learning_rate=1e-5 ae.lat_channels=16
python train_ae.py dataset=euler_all optim.learning_rate=1e-5 ae.lat_channels=32
python train_ae.py dataset=euler_all optim.learning_rate=1e-5 ae.lat_channels=64
python train_ae.py dataset=rayleigh_benard optim.learning_rate=1e-5 ae.lat_channels=16
python train_ae.py dataset=rayleigh_benard optim.learning_rate=1e-5 ae.lat_channels=32
python train_ae.py dataset=rayleigh_benard optim.learning_rate=1e-5 ae.lat_channels=64
```

```
python cache_latents.py dataset=euler_all split=train repeat=4 run=???
python cache_latents.py dataset=euler_all split=valid run=???
python cache_latents.py dataset=rayleigh_benard split=train repeat=4 run=???
python cache_latents.py dataset=rayleigh_benard split=valid run=???
```

### Pixel-space SMs

```
python train_surrogate.py dataset=euler_all optim.learning_rate=1e-5 compute.nodes=2
python train_surrogate.py dataset=rayleigh_benard optim.learning_rate=1e-5
```

### Latent-space DMs and SMs

```
python train_diffusion.py dataset=euler_all ae_run=???
python train_diffusion.py dataset=rayleigh_benard ae_run=???
python train_surrogate.py dataset=euler_all ae_run=???
python train_surrogate.py dataset=rayleigh_benard ae_run=???
```
