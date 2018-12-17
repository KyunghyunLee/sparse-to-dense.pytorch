import numpy as np
import dataloaders.transforms as transforms
import os
import imageio

from dataloaders.dataloader import MyDataloader

iheight, iwidth = 480, 640 # raw image size

DEPTH_LOG = False

class CarlaDataset(MyDataloader):
    def __init__(self, root, type, sparsifier=None, modality='rgbdl', label_type='diluted'):
        self.label_name = ''
        self.label_type = label_type
        super(CarlaDataset, self).__init__(root, type, sparsifier, modality, self.make_dataset, self.carla_loader)
        self.output_size = (150, 200)
        print('Load carla dataset with depth_log:{}'.format(DEPTH_LOG))

    def carla_loader(self, path):
        rgb = imageio.imread(path + '_rgb_raw.png', as_gray=False, pilmode="RGB")
        if DEPTH_LOG:
            depth = imageio.imread(path + '_depth_log.png', as_gray=True).astype('float32') / 255 * 1000.0
        else:
            depth = imageio.imread(path + '_depth_raw.png', as_gray=False, pilmode="RGB").astype('float32')
            depth[:, :, 1] *= 256
            depth[:, :, 2] *= 256 * 256
            depth = np.sum(depth, axis=2) / 16777.215

        return rgb, depth

    def make_dataset(self, dir, class_to_idx):
        images = []
        dir = os.path.expanduser(dir)
        for target in sorted(os.listdir(dir)):
            d = os.path.join(dir, target)
            if not os.path.isdir(d):
                continue
            for root, _, fnames in sorted(os.walk(d)):
                for fname in sorted(fnames):
                    if fname.endswith('rgb_raw.png'):
                        filename = fname[:8]
                        path = os.path.join(root, filename)
                        item = (path, class_to_idx[target])
                        images.append(item)

        return images

    def train_transform_label(self, rgb, depth, label):
        s = np.random.uniform(1.0, 1.5) # random scaling
        depth_np = depth  # / s
        angle = np.random.uniform(-5.0, 5.0) # random rotation degrees
        do_flip = np.random.uniform(0.0, 1.0) < 0.5 # random horizontal flip
        shift_x = np.random.uniform(-50.0, 50.0)

        # perform 1st step of data augmentation
        transform = transforms.Compose([
            transforms.Translate(shift_x, 0.0),
            transforms.Resize(300.0 / iheight), # this is for computational efficiency, since rotation can be slow
            # transforms.Rotate(angle),
            # transforms.Resize(s),
            transforms.CenterCrop(self.output_size),
            transforms.HorizontalFlip(do_flip)
        ])
        label_transform = transforms.Compose([
            transforms.Translate(shift_x / 2.0, 0.0),
            transforms.CenterCrop(self.output_size),
            transforms.HorizontalFlip(do_flip)
        ])
        rgb_np = transform(rgb)
        rgb_np = self.color_jitter(rgb_np) # random color jittering
        rgb_np = np.asfarray(rgb_np, dtype='float') / 255
        depth_np = transform(depth_np)
        label_np = label_transform(label)

        return rgb_np, depth_np, label_np

    def train_transform(self, rgb, depth):
        s = np.random.uniform(1.0, 1.5) # random scaling
        depth_np = depth  # / s
        angle = np.random.uniform(-5.0, 5.0) # random rotation degrees
        do_flip = np.random.uniform(0.0, 1.0) < 0.5 # random horizontal flip

        # perform 1st step of data augmentation
        transform = transforms.Compose([
            transforms.Resize(150.0 / iheight), # this is for computational efficiency, since rotation can be slow
            # transforms.Rotate(angle),
            # transforms.Resize(s),
            transforms.CenterCrop(self.output_size),
            transforms.HorizontalFlip(do_flip)
        ])
        rgb_np = transform(rgb)
        rgb_np = self.color_jitter(rgb_np) # random color jittering
        rgb_np = np.asfarray(rgb_np, dtype='float') / 255
        depth_np = transform(depth_np)

        return rgb_np, depth_np

    def val_transform_label(self, rgb, depth, label):
        depth_np = depth
        transform = transforms.Compose([
            transforms.Resize(300.0 / iheight),
            transforms.CenterCrop(self.output_size),
        ])
        # label_transform = transforms.CenterCrop(self.output_size),

        rgb_np = transform(rgb)
        rgb_np = np.asfarray(rgb_np, dtype='float') / 255
        depth_np = transform(depth_np)
        label_np = label
        return rgb_np, depth_np, label_np

    def val_transform(self, rgb, depth):
        depth_np = depth
        transform = transforms.Compose([
            transforms.Resize(150.0 / iheight),
            # transforms.CenterCrop(self.output_size),
        ])
        rgb_np = transform(rgb)
        rgb_np = np.asfarray(rgb_np, dtype='float') / 255
        depth_np = transform(depth_np)

        return rgb_np, depth_np
