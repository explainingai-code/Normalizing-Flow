import torch
import yaml
import argparse
import os
import numpy as np
import random
from tqdm import tqdm
from torch.optim import AdamW
from dataset.mnist_dataset import MnistDataset
from torch.utils.data import DataLoader
from models.vae import VAE
from models.normalizing_flow import SimpleRealNVP
from tools.sample import sample

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.backends.mps.is_available():
    device = torch.device('mps')
    print('Using mps')


def train(args):
    # Read the config file #
    with open(args.config_path, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)
    print(config)
    ########################

    dataset_config = config['dataset_params']
    normflow_config = config['normflow_params']
    autoencoder_model_config = config['autoencoder_params']
    train_config = config['train_params']

    # Set the desired seed value #
    seed = train_config['seed']
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if device == 'cuda':
        torch.cuda.manual_seed_all(seed)
    if device == 'mps':
        torch.mps.manual_seed(seed)
    #############################

    im_dataset = MnistDataset(split='train',im_path=dataset_config['im_path'],
                              latent_path=os.path.join(train_config['task_name'],
                                                       train_config['vae_latent_dir_name']))

    data_loader = DataLoader(im_dataset,
                             batch_size=train_config['norm_flow_batch_size'],
                             shuffle=True)

    # Instantiate the model

    model = SimpleRealNVP(config).to(device)
    model.train()

    # Create output directories
    if not os.path.exists(train_config['task_name']):
        os.mkdir(train_config['task_name'])

    if os.path.exists(os.path.join(train_config['task_name'],
                                   train_config['normflow_ckpt_name'])):
        print('Loaded RealNVP checkpoint')
        model.load_state_dict(torch.load(os.path.join(train_config['task_name'],
                                                      train_config['normflow_ckpt_name']),
                                         map_location=device))

    vae = VAE(im_channels=dataset_config['im_channels'],
              model_config=autoencoder_model_config).to(device)
    vae.eval()

    assert os.path.exists(os.path.join(train_config['task_name'], train_config[
        'vae_autoencoder_ckpt_name'])), "VAE checkpoint not present. Train VAE first."
    vae.load_state_dict(torch.load(os.path.join(train_config['task_name'],
                                                train_config['vae_autoencoder_ckpt_name']),
                                   map_location=device), strict=True)
    print('Loaded vae checkpoint')

    num_epochs = train_config['normflow_epochs']

    scale_params = [param for name, param in model.named_parameters() if 'log_scale' in name]
    other_params = [param for name, param in model.named_parameters() if 'log_scale' not in name]

    optimizer = AdamW([
        {'params': scale_params, 'weight_decay': 1E-5},
        {'params': other_params, 'weight_decay': 0}
    ], lr=train_config['normflow_lr'])

    step_count = 0
    for epoch_idx in range(num_epochs):
        losses = []
        for im in tqdm(data_loader):
            optimizer.zero_grad()
            im = im.float().to(device)
            mean, logvar = torch.chunk(im, 2, dim=1)
            std = torch.exp(0.5 * logvar)
            im = mean + std * torch.randn(mean.shape).to(device=im.device)

            bs = im.shape[0]
            if not normflow_config['conv']:
                im = im.reshape(bs, -1)

            x, log_prob, log_jacobian_determinants = model(im)
            loss = log_jacobian_determinants + log_prob

            loss = (-loss).mean()
            losses.append(loss.item())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
            optimizer.step()
            if step_count % train_config['normflow_img_save_steps'] == 0:
                sample(model, config, vae)
            step_count += 1
        print('Finished epoch:{} | Loss : {:.4f}'.format(
            epoch_idx + 1,
            np.mean(losses)))

        torch.save(model.state_dict(), os.path.join(train_config['task_name'],
                                                    train_config['normflow_ckpt_name']))

    print('Done Training ...')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Arguments for normalizing flow training')
    parser.add_argument('--config', dest='config_path',
                        default='config/mnist.yaml', type=str)
    args = parser.parse_args()
    train(args)




