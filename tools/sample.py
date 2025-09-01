import torch
import torchvision
import argparse
import yaml
import os
from torchvision.utils import make_grid
import math
from models.vae import VAE
from models.normalizing_flow import SimpleRealNVP

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.backends.mps.is_available():
    device = torch.device('mps')
    print('Using mps')


def sample(model, config, vae):
    num_samples = config['train_params']['num_samples']
    im_latent_size = config['dataset_params']['im_size'] // 2 ** sum(
            config['autoencoder_params']['down_sample'])
    z_channels = config['autoencoder_params']['z_channels']
    x = torch.randn((num_samples, z_channels, im_latent_size, im_latent_size)).to(device)
    if not config['normflow_params']['conv']:
        x = x.reshape((num_samples, -1))
    x = model.inverse(x)
    x = x.reshape((num_samples, z_channels, im_latent_size, im_latent_size))
    x = vae.to(device).decode(x)
    x = torch.clamp(x, -1, 1.)
    x = (x + 1) / 2

    x_grid = make_grid(x.cpu(), nrow=int(math.sqrt(num_samples)))
    x_grid = torchvision.transforms.ToPILImage()(x_grid)

    if not os.path.exists(os.path.join(config['train_params']['task_name'], 'samples')):
        os.mkdir(os.path.join(config['train_params']['task_name'], 'samples'))
    x_grid.save(os.path.join(config['train_params']['task_name'], 'samples.png'))
    x_grid.close()


def infer(args):
    # Read the config file #
    with open(args.config_path, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)
    print(config)
    ########################

    dataset_config = config['dataset_params']
    autoencoder_model_config = config['autoencoder_params']
    train_config = config['train_params']

    model = SimpleRealNVP(config).to(device)
    model.eval()
    assert os.path.exists(os.path.join(train_config['task_name'],
                                   train_config['normflow_ckpt_name'])), \
        "RealNVP checkpoint not present. Train normalizing flows model first."
    print('Loaded RealNVP checkpoint')
    model.load_state_dict(torch.load(os.path.join(train_config['task_name'],
                                                  train_config['normflow_ckpt_name']),
                                     map_location=device))
    # Create output directories
    if not os.path.exists(train_config['task_name']):
        os.mkdir(train_config['task_name'])

    vae = VAE(im_channels=dataset_config['im_channels'],
              model_config=autoencoder_model_config).to(device)
    vae.eval()

    # Load vae if found
    assert os.path.exists(os.path.join(train_config['task_name'],
                                       train_config['vae_autoencoder_ckpt_name'])), \
        "VAE checkpoint not present. Train VAE first."
    vae.load_state_dict(torch.load(os.path.join(train_config['task_name'],
                                                train_config['vae_autoencoder_ckpt_name']),
                                   map_location=device), strict=True)
    print('Loaded vae checkpoint')

    with torch.no_grad():
        sample(model, config, vae)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Arguments for normalizing flow generation')
    parser.add_argument('--config', dest='config_path',
                        default='config/mnist.yaml', type=str)
    args = parser.parse_args()
    infer(args)
