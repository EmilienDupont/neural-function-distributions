# Generative Models as Distributions of Functions

This repo contains code to reproduce all experiments in [Generative Models as Distributions of Functions](https://arxiv.org/abs/2102.04776).

<img src="https://github.com/EmilienDupont/neural-function-distributions/raw/main/imgs/example.gif" height="200"> <img src="https://github.com/EmilienDupont/neural-function-distributions/raw/main/imgs/example.gif" height="200">

## Requirements

Requirements for training the models can be installed using `pip install -r requirements.txt`. All experiments were run using `python 3.8.10`.

## Training a model

To train a model on CelebAHQ64, run

```python main.py configs/config_celebahq64.json```

Example configs to reproduce the results in the paper are provided in the `configs` folder. Note that you will have to provide a path to the data you wish to train on in the config.

## Downloading datasets

The shapenet and ERA5 climate datasets can be downloaded at this [link](https://drive.google.com/drive/folders/1r_sk5auYvllSpDG9ZjroOG0SH0v5kPmM?usp=sharing). The CelebAHQ datasets can be downloaded from [here](https://drive.google.com/drive/folders/11Vz0fqHS2rXDb5pprgTjpD7S2BAJhi1P).

## Loading trained models

All trained models can be downloaded from [here](https://drive.google.com/drive/folders/1r_sk5auYvllSpDG9ZjroOG0SH0v5kPmM?usp=sharing). The `load-trained-model.ipynb` notebook shows an example of using a trained model.

## Rendering 3D samples

The requirements in `requirements.txt` allow for basic plotting of 3D shapes with matplotlib. However, to properly render 3D models, you will need to install [mcubes](https://github.com/pmneila/PyMCubes) (for marching cubes) and [pytorch3d](https://github.com/facebookresearch/pytorch3d). Pytorch3D is not directly pip installable (depending on your version of torch), so please follow the [install instructions](https://github.com/facebookresearch/pytorch3d/blob/master/INSTALL.md) provided in their repo.

Once these requirements have been installed, you can render samples using

```python render_mesh.py -sp samples -mr 32 -rs 64 -ns 8 -mp trained-models/shapenet_voxels/model.pt```

See `render_mesh.py` for a full list of rendering options.

## Globe plots

By default the samples from the climate data will be visualized as flat projections. Plotting these on a globe (using the `viz/plots_globe.py` functions) requires the [cartopy](https://scitools.org.uk/cartopy/docs/latest/) library. Installation instructions for this library can be found [here](https://scitools.org.uk/cartopy/docs/latest/installing.html).

## License

MIT
