Normalizing Flows Implementation in PyTorch
========

This repository implements a Normalizing Flows model, for which we use a simple real nvp like model.
The repo trains on mnist dataset but rather training on images in pixel space we first use an autoencoder and train 
normalizing flows model on latent images.
As of today the repo provides code to do the following:
* Training and Inference of a Normalizing flows model(similar to realnvp) on latent mnist images
  * For this the repo provides both real nvp with linear layers as well as convolutional layers
* Training and Inference of a VAE trained with perceptual loss on mnist dataset


## Normalizing Flows Tutorial Video
<a href="https://www.youtube.com/watch?v=x11yFPsttRw">
   <img alt="Normalizing Flows Tutorial" src="https://github.com/user-attachments/assets/d4800d9b-7cfa-4006-9510-7368f129eb65"
   width="400">
</a>
___  

## Sample Output for trained normalizing flows model on mnist
Linear Model - Left, Convolutional Model - Right

<img width="300" alt="linear-normflow-output" src="https://github.com/user-attachments/assets/be157820-5a79-4254-87e7-4977bf38b113" />
<img width="300" alt="conv-normflow-output" src="https://github.com/user-attachments/assets/3c7b82d2-5168-4c7e-a00c-c49b7d4dd4ec" />

## Sample Output for Autoencoder on MNIST
Image - Top, Reconstructions - Below

<img src="https://github.com/user-attachments/assets/1be2cf25-4488-48d0-9068-d80c23c65854" width="200">

___

## Setup
* Create a new conda environment with python 3.10 then run below commands
* ```git clone https://github.com/explainingai-code/NormalizingFlow-PyTorch.git```
* ```cd NormalizingFlow-PyTorch```
* ```pip install -r requirements.txt```
* Download lpips weights by opening this link in browser(dont use cURL or wget) https://github.com/richzhang/PerceptualSimilarity/blob/master/lpips/weights/v0.1/vgg.pth and downloading the raw file. Place the downloaded weights file in `models/weights/v0.1/vgg.pth`

___  

## Data Preparation
### Mnist

For setting up the mnist dataset follow - https://github.com/explainingai-code/Pytorch-VAE#data-preparation

Ensure directory structure is following
```
NormalizingFlow-PyTorch
    -> data
        -> mnist
            -> train
                -> images
                    -> *.png
            -> test
                -> images
                    -> *.png
```

---
## Configuration
 Allows you to play with different components of normalizing flows and autoencoder training
* ```config/mnist.yaml``` - Linear normalizing flows model
* ```config/mnist_conv.yaml``` - Convolutional normalizing flows model

___  
## Training
The repo provides training and inference for Mnist but for working on your own dataset:
* Create your own config and have the path in config point to images (look at `mnist.yaml` for guidance)
* Create your own dataset class which will just collect all the filenames and return the image in its getitem method(for autoencoder) and saved latents(for normalizing flows). Look at `mnist_dataset.py` for guidance 

Once the config and dataset is setup:
* Train the auto encoder on your dataset using [this section](#training-autoencoder-for-mnist)
* For training Normalizing Flows model follow [this section](#training-normalizing-flows-model)


## Training AutoEncoder for MNIST
* For training autoencoder on mnist,ensure the right path is mentioned in `mnist.yaml`
* For training autoencoder on your own dataset 
  * Create your own config and have the path point to images (look at mnist.yaml for guidance)
  * Create your own dataset class, similar to mnist_dataset.py 
  * Use the new dataset class [here](https://github.com/explainingai-code/Normalizing-Flow/blob/main/tools/train_vae.py#L50)
* For training autoencoder run ```python -m tools.train_vae --config config/mnist.yaml``` for training vae with the desire config file
* For inference using trained autoencoder run```python -m tools.infer_vae --config config/mnist.yaml``` for generating reconstructions with right config file. Use `save_latent=True` in config to save the latent files 


## Training Normalizing Flows Model
Train the autoencoder first and setup dataset accordingly.
* ```python -m tools.train --config config/mnist.yaml``` for training normalizing flows model using linear layers. Use config/mnist_conv.yaml for convolutional model.
* ```python -m tools.sample --config config/mnist.yaml``` for sampling from normalizing flows model using linear layers. Use config/mnist_conv.yaml for convolutional model.



## Output 
Outputs will be saved according to the configuration present in yaml files.

For every run a folder of ```task_name``` key in config will be created

During training of autoencoder the following output will be saved 
* Latest Autoencoder and discriminator checkpoint in ```task_name``` directory
* Sample reconstructions in ```task_name/vae_autoencoder_samples```

During inference of autoencoder the following output will be saved
* Reconstructions for random images in  ```task_name```
* Latents will be save in ```task_name/vae_latent_dir_name``` if mentioned in config

During training and inference of normalizing flows model, following output will be saved
* During training of normalizing flows model, we will save the latest checkpoint in ```task_name``` directory
* During sampling, sampled image grid will be saved in  ```task_name/samples.png``` . 



