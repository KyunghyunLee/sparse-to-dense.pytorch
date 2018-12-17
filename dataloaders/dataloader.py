import os
import os.path
import numpy as np
import torch.utils.data as data
import h5py
import dataloaders.transforms as transforms

import imageio
import cv2
from matplotlib import pyplot as plt

IMG_EXTENSIONS = ['.h5',]

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def find_classes(dir):
    classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx

def make_dataset(dir, class_to_idx):
    images = []
    dir = os.path.expanduser(dir)
    for target in sorted(os.listdir(dir)):
        d = os.path.join(dir, target)
        if not os.path.isdir(d):
            continue
        for root, _, fnames in sorted(os.walk(d)):
            for fname in sorted(fnames):
                if is_image_file(fname):
                    path = os.path.join(root, fname)
                    item = (path, class_to_idx[target])
                    images.append(item)
    return images

def h5_loader(path):
    h5f = h5py.File(path, "r")
    rgb = np.array(h5f['rgb'])
    rgb = np.transpose(rgb, (1, 2, 0))
    depth = np.array(h5f['depth'])
    return rgb, depth

# def rgb2grayscale(rgb):
#     return rgb[:,:,0] * 0.2989 + rgb[:,:,1] * 0.587 + rgb[:,:,2] * 0.114

to_tensor = transforms.ToTensor()

class MyDataloader(data.Dataset):
    modality_names = ['rgb', 'rgbd', 'd', 'rgbl', 'rgbde'] # , 'g', 'gd'
    color_jitter = transforms.ColorJitter(0.4, 0.4, 0.4)

    def __init__(self, root, type, sparsifier=None, modality='rgb', make_dataset=make_dataset, loader=h5_loader):
        classes, class_to_idx = find_classes(root)
        imgs = make_dataset(root, class_to_idx)
        assert len(imgs)>0, "Found 0 images in subfolders of: " + root + "\n"
        print("Found {} images in {} folder.".format(len(imgs), type))
        self.root = root
        self.imgs = imgs
        self.classes = classes
        self.class_to_idx = class_to_idx
        if type == 'train':
            if modality == 'rgbl':
                self.transform = self.train_transform_label
            else:
                self.transform = self.train_transform
        elif type == 'val':
            if modality == 'rgbl':
                self.transform = self.val_transform_label
            else:
                self.transform = self.val_transform
        else:
            raise (RuntimeError("Invalid dataset type: " + type + "\n"
                                "Supported dataset types are: train, val"))
        self.loader = loader
        self.sparsifier = sparsifier

        assert (modality in self.modality_names), "Invalid modality type: " + modality + "\n" + \
                                "Supported dataset types are: " + ''.join(self.modality_names)
        self.modality = modality

    def train_transform_label(self, rgb, depth, label):
        raise (RuntimeError("train_transform() is not implemented. "))

    def val_transform_label(self, rgb, depth, label):
        raise (RuntimeError("val_transform() is not implemented."))

    def train_transform(self, rgb, depth):
        raise (RuntimeError("train_transform() is not implemented. "))

    def val_transform(self, rgb, depth):
        raise (RuntimeError("val_transform() is not implemented."))

    def create_sparse_depth(self, rgb, depth):
        if self.sparsifier is None:
            return depth
        else:
            mask_keep = self.sparsifier.dense_to_sparse(rgb, depth)
            sparse_depth = np.zeros(depth.shape)
            sparse_depth[mask_keep] = depth[mask_keep]
            return sparse_depth

    def create_rgbd(self, rgb, depth):
        sparse_depth = self.create_sparse_depth(rgb, depth)
        rgbd = np.append(rgb, np.expand_dims(sparse_depth, axis=2), axis=2)
        return rgbd

    def create_rgbde(self, rgb, depth):
        sparse_depth = self.create_sparse_depth(rgb, depth)
        edge = cv2.Canny(rgb, 100, 200)
        rgbd = np.append(rgb, np.expand_dims(sparse_depth, axis=2), axis=2)
        rgbde = np.append(rgbd, np.expand_dims(edge, axis=2), axis=2)
        return rgbde

    def create_rgbdl(self, rgb, depth):
        rgbd = np.append(rgb, np.expand_dims(depth, axis=2), axis=2)
        return rgbd

    def get_label(self, index):
        path, target = self.imgs[index]
        filename = path + '_' + self.label_type + '_label.png'
        label = imageio.imread(filename).astype('float32') / 255.0 * 1000.0
        return label

    def __getraw__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (rgb, depth) the raw data.
        """
        path, target = self.imgs[index]
        rgb, depth = self.loader(path)
        return rgb, depth

    def __getitem__(self, index):
        rgb, depth = self.__getraw__(index)
        if self.modality != 'rgbl':
            if self.transform is not None:
                rgb_np, depth_np = self.transform(rgb, depth)
            else:
                raise(RuntimeError("transform not defined"))

        # color normalization
        # rgb_tensor = normalize_rgb(rgb_tensor)
        # rgb_np = normalize_np(rgb_np)
        label_np = None
        debug_figure = False
        if debug_figure:
            fig = plt.figure(1)
            a1 = fig.add_subplot(2, 3, 1)
            a2 = fig.add_subplot(2, 3, 2)
            a3 = fig.add_subplot(2, 3, 3)
            a4 = fig.add_subplot(2, 3, 4)
            a5 = fig.add_subplot(2, 3, 5)
            a6 = fig.add_subplot(2, 3, 6)

        if self.modality == 'rgb':
            input_np = rgb_np
        elif self.modality == 'rgbd':
            input_np = self.create_rgbd(rgb_np, depth_np)
        elif self.modality == 'd':
            input_np = self.create_sparse_depth(rgb_np, depth_np)
        elif self.modality == 'rgbde':
            input_np = self.create_rgbde(rgb_np, depth_np)
        elif self.modality == 'rgbl':
            label_np = self.get_label(index)
            if debug_figure:
                a1.imshow(rgb)
                a2.imshow(depth)
                a3.imshow(label_np)
            rgb_np, depth_np, label_np = self.transform(rgb, depth, label_np)
            input_np = self.create_rgbdl(rgb_np, depth_np)

        if debug_figure:
            a4.imshow(rgb_np)
            a5.imshow(depth_np)
            a6.imshow(label_np)
            plt.show()

        input_tensor = to_tensor(input_np)
        while input_tensor.dim() < 3:
            input_tensor = input_tensor.unsqueeze(0)
        if label_np is not None:
            label_tensor = to_tensor(label_np)
        else:
            label_tensor = to_tensor(depth_np)
        label_tensor = label_tensor.unsqueeze(0)

        return input_tensor, label_tensor

    def __len__(self):
        return len(self.imgs)

    # def __get_all_item__(self, index):
    #     """
    #     Args:
    #         index (int): Index

    #     Returns:
    #         tuple: (input_tensor, depth_tensor, input_np, depth_np)
    #     """
    #     rgb, depth = self.__getraw__(index)
    #     if self.transform is not None:
    #         rgb_np, depth_np = self.transform(rgb, depth)
    #     else:
    #         raise(RuntimeError("transform not defined"))

    #     # color normalization
    #     # rgb_tensor = normalize_rgb(rgb_tensor)
    #     # rgb_np = normalize_np(rgb_np)

    #     if self.modality == 'rgb':
    #         input_np = rgb_np
    #     elif self.modality == 'rgbd':
    #         input_np = self.create_rgbd(rgb_np, depth_np)
    #     elif self.modality == 'd':
    #         input_np = self.create_sparse_depth(rgb_np, depth_np)

    #     input_tensor = to_tensor(input_np)
    #     while input_tensor.dim() < 3:
    #         input_tensor = input_tensor.unsqueeze(0)
    #     depth_tensor = to_tensor(depth_np)
    #     depth_tensor = depth_tensor.unsqueeze(0)

    #     return input_tensor, depth_tensor, input_np, depth_np
