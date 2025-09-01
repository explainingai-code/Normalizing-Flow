import glob
import os
import pickle
import torchvision
from tqdm import tqdm
from PIL import Image

from torch.utils.data.dataset import Dataset


class MnistDataset(Dataset):
    def __init__(self, split, im_path, latent_path=None, im_ext='png', im_size=28):
        r"""
        Init method for initializing the dataset properties
        :param split: train/test to locate the image files
        :param im_path: root folder of images
        :param im_ext: image extension. assumes all
        images would be this type.
        """
        self.split = split
        self.im_size = im_size
        self.im_ext = im_ext
        self.use_latents = latent_path is not None
        self.images = self.load_images(im_path)
        if self.use_latents:
            self.latent_maps = self.load_latents(latent_path)
            assert len(self.latent_maps) == len(self.images), \
                ("Latents not saved for all images. Number of latents = {} but number of images = {}".
                 format(len(self.latent_maps), len(self.images)))

    def load_latents(self, latent_path):
        latent_maps = {}
        for fname in glob.glob(os.path.join(latent_path, '*.pkl')):
            s = pickle.load(open(fname, 'rb'))
            for k, v in s.items():
                latent_maps[k] = v[0]
        return latent_maps

    def load_images(self, im_path):
        assert os.path.exists(im_path), "images path {} does not exist".format(im_path)
        ims = []
        labels = []
        for d_name in tqdm(os.listdir(im_path)):
            for fname in glob.glob(os.path.join(im_path, d_name, '*.{}'.format(self.im_ext))):
                ims.append(fname)
        print('Found {} images for split {}'.format(len(ims), self.split))
        return ims

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        im_path = self.images[index]
        if self.use_latents:
            latent = self.latent_maps[im_path]
            return latent
        else:
            im = Image.open(im_path)
            im_tensor = torchvision.transforms.ToTensor()(im)

            # Convert input to -1 to 1 range.
            im_tensor = (2 * im_tensor) - 1
            return im_tensor

