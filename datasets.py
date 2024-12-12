import os
import torch
from torchvision import datasets

import pathlib
from typing import Any, Callable, Optional, Sequence, Tuple, Union
from torchvision.datasets.vision import VisionDataset
from torchvision.datasets.utils import download_and_extract_archive, verify_str_arg
from torch.utils.data import DataLoader, Dataset
from PIL import Image

class CUBDataset(datasets.ImageFolder):
    """
    Wrapper for the CUB-200-2011 dataset. 
    Method DatasetBirds.__getitem__() returns tuple of image and its corresponding label.    
    Dataset per https://github.com/slipnitskaya/caltech-birds-advanced-classification
    """
    def __init__(self,
                 root,
                 transform=None,
                 target_transform=None,
                 loader=datasets.folder.default_loader,
                 is_valid_file=None,
                 train=True,
                 bboxes=False):

        img_root = os.path.join(root, 'images')

        super(CUBDataset, self).__init__(
            root=img_root,
            transform=None,
            target_transform=None,
            loader=loader,
            is_valid_file=is_valid_file,
        )
        
        self.redefine_class_to_idx()

        self.transform_ = transform
        self.target_transform_ = target_transform
        self.train = train
        
        # obtain sample ids filtered by split
        path_to_splits = os.path.join(root, 'train_test_split.txt')
        indices_to_use = list()
        with open(path_to_splits, 'r') as in_file:
            for line in in_file:
                idx, use_train = line.strip('\n').split(' ', 2)
                if bool(int(use_train)) == self.train:
                    indices_to_use.append(int(idx))

        # obtain filenames of images
        path_to_index = os.path.join(root, 'images.txt')
        filenames_to_use = set()
        with open(path_to_index, 'r') as in_file:
            for line in in_file:
                idx, fn = line.strip('\n').split(' ', 2)
                if int(idx) in indices_to_use:
                    filenames_to_use.add(fn)

        img_paths_cut = {'/'.join(img_path.rsplit('/', 2)[-2:]): idx for idx, (img_path, lb) in enumerate(self.imgs)}
        imgs_to_use = [self.imgs[img_paths_cut[fn]] for fn in filenames_to_use]

        _, targets_to_use = list(zip(*imgs_to_use))

        self.imgs = self.samples = imgs_to_use
        self.targets = targets_to_use

        if bboxes:
            # get coordinates of a bounding box
            path_to_bboxes = os.path.join(root, 'bounding_boxes.txt')
            bounding_boxes = list()
            with open(path_to_bboxes, 'r') as in_file:
                for line in in_file:
                    idx, x, y, w, h = map(lambda x: float(x), line.strip('\n').split(' '))
                    if int(idx) in indices_to_use:
                        bounding_boxes.append((x, y, w, h))

            self.bboxes = bounding_boxes
        else:
            self.bboxes = None

    def __getitem__(self, index):
        # generate one sample
        sample, target = super(CUBDataset, self).__getitem__(index)

        if self.bboxes is not None:
            # squeeze coordinates of the bounding box to range [0, 1]
            width, height = sample.width, sample.height
            x, y, w, h = self.bboxes[index]

            scale_resize = 500 / width
            scale_resize_crop = scale_resize * (375 / 500)

            x_rel = scale_resize_crop * x / 375
            y_rel = scale_resize_crop * y / 375
            w_rel = scale_resize_crop * w / 375
            h_rel = scale_resize_crop * h / 375

            target = torch.tensor([target, x_rel, y_rel, w_rel, h_rel])

        if self.transform_ is not None:
            sample = self.transform_(sample)
        if self.target_transform_ is not None:
            target = self.target_transform_(target)

        return sample, target
    
    def redefine_class_to_idx(self):
        adjusted_dict = {}
        for k, v in self.class_to_idx.items():
            k = k.split('.')[-1].replace('_', ' ')
            split_key = k.split(' ')
            if len(split_key) > 2: 
                k = '-'.join(split_key[:-1]) + " " + split_key[-1]
            adjusted_dict[k] = v
        self.class_to_idx = adjusted_dict
                
                

from PIL import Image
import torchvision.transforms as transforms

def _transform(n_px):
    return transforms.Compose([
        transforms.Resize(n_px, interpolation=Image.BICUBIC),
        transforms.CenterCrop(n_px),
        lambda image: image.convert("RGB"),
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])

def _transform_origin(n_px):
    return transforms.Compose([
        transforms.Resize(n_px, interpolation=Image.BICUBIC),
        transforms.CenterCrop(n_px),
        lambda image: image.convert("RGB"),
        transforms.ToTensor(),
    ])



class RemappedImageFolder(datasets.ImageFolder):
    def __init__(self, root, transform=None, target_transform=None, loader=datasets.folder.default_loader, is_valid_file=None, remapping=None):
        super(RemappedImageFolder, self).__init__(root, transform=transform, target_transform=target_transform, loader=loader, is_valid_file=is_valid_file)
        self.remapping = remapping

    def __getitem__(self, index):
        path, target = self.samples[index]
        if self.remapping is not None and target in self.remapping:
            target = self.remapping[target]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return sample, target



class OxfordIIITPet(VisionDataset):
    """
    Oxford-IIIT Pet Dataset with caption support.

    Args:
        root (str or pathlib.Path): Root directory of the dataset.
        split (string, optional): The dataset split, supports "trainval" or "test".
        target_types (sequence of strings, optional): Types of target to use. Can be "category" or
            "segmentation". Can also be a list to output a tuple with all specified target types.
        transform (callable, optional): A function/transform that takes in a PIL image and returns a transformed version.
        target_transform (callable, optional): A function/transform that takes in the target and transforms it.
        download (bool, optional): If True, downloads the dataset from the internet and puts it into root/oxford-iiit-pet.
        captions_file (str, optional): Filename of the captions file.
    """

    _RESOURCES = (
        ("https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz", "5c4f3ee8e5d25df40f4fd59a7f44e54c"),
        ("https://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz", "95a8c909bbe2e81eed6a22bccdf3f68f"),
    )
    _VALID_TARGET_TYPES = ("category", "segmentation")

    def __init__(
        self,
        root: Union[str, pathlib.Path],
        split: str = "trainval",
        target_types: Union[Sequence[str], str] = "category",
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
        captions_file: str = "captions.txt",
    ):
        super().__init__(root, transform=transform, target_transform=target_transform)
        self.root = root
        self.split = verify_str_arg(split, "split", ("trainval", "test"))
        self.target_types = target_types if isinstance(target_types, list) else [target_types]
        self.captions_file = captions_file

        self._base_folder = pathlib.Path(self.root) / "oxford-iiit-pet"
        self._images_folder = self._base_folder / "images"
        self._anns_folder = self._base_folder / "annotations"
        self._segs_folder = self._anns_folder / "trimaps"

        if download:
            self._download()

        if not self._check_exists():
            raise RuntimeError("Dataset not found. You can use download=True to download it")

        self._load_metadata()

    def _load_metadata(self):
        # Load captions
        self.captions = {}
        captions_path = self._base_folder / self.captions_file
        if os.path.exists(captions_path):
            with open(captions_path, 'r') as f:
                for line in f:
                    image_file, caption = line.strip().split(': ', 1)
                    self.captions[image_file] = caption

        # Load image and annotation information
        self.images = []
        self.labels = []
        self.segments = []
        with open(self._anns_folder / f"{self.split}.txt") as f:
            for line in f:
                image_id, label = line.strip().split()[:2]
                image_file = f"{image_id}.jpg"
                self.images.append(self._images_folder / image_file)
                self.labels.append(int(label) - 1)
                self.segments.append(self._segs_folder / f"{image_id}.png")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = Image.open(self.images[idx]).convert("RGB")

        target = []
        for target_type in self.target_types:
            if target_type == "category":
                target.append(self.labels[idx])
            elif target_type == "segmentation":
                target.append(Image.open(self.segments[idx]))

        if not target:
            target = None
        elif len(target) == 1:
            target = target[0]
        else:
            target = tuple(target)

        caption = self.captions.get(os.path.basename(self.images[idx]), "")

        if self.transform:
            image = self.transform(image)

        return image, target, caption

    def _check_exists(self) -> bool:
        return os.path.exists(self._images_folder) and os.path.exists(self._anns_folder)

    def _download(self):
        if self._check_exists():
            return

        for url, md5 in self._RESOURCES:
            download_and_extract_archive(url, download_root=str(self._base_folder), md5=md5)


class CocoCaptionsDataset(Dataset):
    def __init__(self, root, annFile, transform):
        self.dataset = datasets.CocoCaptions(root=root, annFile=annFile, transform=transform)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, captions = self.dataset[idx]
        # Select the first caption or another strategy to select one caption
        caption = captions[0]
        return image, caption  # Return the first tokenized caption


class Flickr30kCaptionsDataset(Dataset):
    def __init__(self, root, ann_file, transform):
        self.dataset = datasets.Flickr30k(root=root, ann_file=ann_file, transform=transform)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, captions = self.dataset[idx]
        # Select the first caption or another strategy to select one caption
        caption = captions[0]
        return image, caption  # Return the first tokenized caption