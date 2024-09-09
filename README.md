# MPP x Latent Diffusion Models

## Code

To run the experiments, it is necessary to have access to a [Slurm](https://slurm.schedmd.com) cluster, to export a [Neptune](https://neptune.ai) API token (in `~/.bash_profile`) and to install the [lpdm](lpdm) module as a package. First, create a new Python environment, for example with [venv](https://docs.python.org/3/library/venv.html).

```
python -m venv ~/.venvs/ldpm
source ~/.venvs/lpdm/bin/activate
```

Then, install the [ldpm](ldpm) module as an [editable](https://pip.pypa.io/en/latest/topics/local-project-installs) package with its dependencies.

```
pip install --editable .[all] --extra-index-url https://download.pytorch.org/whl/cu121
```

Optionally, we provide [pre-commit hooks](pre-commit.yml) to automatically detect code issues.

```
pre-commit install --config pre-commit.yaml
```
