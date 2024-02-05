# ---------------------------------------------------------------
# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# for I2SB. To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------

import os, sys
import functools
import io

from PIL import Image
import lmdb

import torch
import torchvision.datasets as datasets
from torchvision import transforms
from torch.utils.data import Dataset

from ipdb import set_trace as debug
sys.path.append("../")
#from dataset.LMDB2ImageFolder import ImageFolderLMDB

def lmdb_loader(path, lmdb_data):
    # In-memory binary streams
    # with lmdb_data.begin(write=False, buffers=True) as txn:
    #     keys = list(txn.cursor().iternext(values=False))
    with lmdb_data.begin(write=False, buffers=True) as txn:
        bytedata = txn.get(path.encode())
        #cursor = txn.cursor()
        #bytedata = cursor.value()
        #bytedata = txn.get(keys[0])
    img = Image.open(io.BytesIO(bytedata))
    return img.convert('RGB')

#[TODO] replace lambda function in L:74 with a function to avoid incompatible issue with pickle in multiprocessing
def _set_loader(path, lmdb_loader, lmdb_env):
    return lmdb_loader(path, lmdb_env)

def _build_lmdb_dataset(
         root, log, transform=None, target_transform=None,
        loader=lmdb_loader):
    """
    You can create this dataloader using:
    train_data = _build_lmdb_dataset(traindir, transform=train_transform)
    valid_data = _build_lmdb_dataset(validdir, transform=val_transform)
    """

    root = str(root)
    if root.endswith('/'):
        root = root[:-1]
    pt_path = os.path.join(
        root + '_faster_imagefolder.lmdb.pt')
    lmdb_path = os.path.join(
        root + '_faster_imagefolder.lmdb')

    if os.path.isfile(pt_path) and os.path.isdir(lmdb_path): # when lmdb is created and saved to disk
        log.info('[Dataset] Loading pt {} and lmdb {}'.format(pt_path, lmdb_path))
        data_set = torch.load(pt_path)
    else:
        data_set = datasets.ImageFolder(
            root, None, None, None)
        torch.save(data_set, pt_path, pickle_protocol=4)
        log.info('[Dataset] Saving pt to {}'.format(pt_path))
        log.info('[Dataset] Building lmdb to {}'.format(lmdb_path))
        #env = lmdb.open(lmdb_path, map_size=1.5e12) # changed to 1.5e12 > 160GB of dataset
        env = lmdb.open(lmdb_path, subdir=True, map_size=1099511627776 * 2, readonly=False, meminit=False, map_async=True) # changed for solving error
        # writting to LMDB file in 'ascii' encode (b'filename')
        with env.begin(write=True) as txn:
            for _path, class_index in data_set.imgs: # _path: filename, 
                with open(_path, 'rb') as f:
                    data = f.read()
                txn.put(_path.encode('ascii'), data) # txn.put(key, value)
    data_set.lmdb_data = lmdb.open(
        lmdb_path, readonly=True, max_readers=1, lock=False, readahead=False,
        meminit=False)
    with data_set.lmdb_data.begin() as txn:
        length = txn.stat()['entries']
    print('\033[91m' + str(length)+ '\033[0m') # checking if LMDB actually loaded
    # reset transform and target_transform
    data_set.samples = data_set.imgs
    data_set.transform = transform 
    data_set.target_transform = target_transform
    # self.loader is used in __getitem__ in torchvision/dataset/folder
    data_set.loader = lambda path: loader(path, data_set.lmdb_data) # parse lmdb_loader(path, lmdb_data)
    #data_set.loader = ImageFolderLMDB(root, data_set.transform) # parse lmdb_loader(path, lmdb_data)
    #data_set.loader = functools.partial(_set_loader, lmdb_loader=loader, lmdb_env=data_set.lmdb_data) # parse lmdb_loader(path, lmdb_data)

    return data_set

#[FIXME]:
# 1. no wrap function or nested function definitions, which is not picklble
def lambda_fn(x):
    return (x*2)-1
def build_train_transform(image_size):
    return transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        #transforms.Lambda(lambda t: (t * 2) - 1) # [0,1] --> [-1, 1]
        transforms.Lambda(lambda_fn)
    ])

def build_test_transform(image_size):
    return transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        # transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        #transforms.Lambda(lambda t: (t * 2) - 1) # [0,1] --> [-1, 1]
        transforms.Lambda(lambda_fn)
    ])

def build_lmdb_dataset(opt, log, train, transform=None):
    """ resize -> crop -> to_tensor -> norm(-1,1) """
    root_folder = opt.dataset_dir / ('train' if train else 'val')

    if transform is None:
        build_transform = build_train_transform if train else build_test_transform
        transform = build_transform(opt.image_size)

    dataset = _build_lmdb_dataset(root_folder, log, transform=transform)
    log.info(f"[Dataset] Built Imagenet dataset {root_folder=}, size={len(dataset)}!")
    return dataset

def readlines(fn):
    file = open(fn, "r").readlines()
    return [line.strip('\n\r') for line in file]

def build_lmdb_dataset_val10k(opt, log, transform=None):

    fn_10k = readlines(f"dataset/val_faster_imagefolder_10k_fn.txt")
    label_10k = readlines(f"dataset/val_faster_imagefolder_10k_label.txt")

    if transform is None: transform = build_test_transform(opt.image_size)
    dataset = _build_lmdb_dataset(opt.dataset_dir / 'val', log, transform=transform) # FIXME why using 10k val?
    dataset.samples = [(fn, int(label)) for fn, label in zip(fn_10k, label_10k)]

    assert len(dataset) == 10_000
    log.info(f"[Dataset] Built Imagenet val10k, size={len(dataset)}!")
    return dataset

def build_lmdb_dataset_val_custom(opt, log, transform=None):

    if transform is None: transform = build_test_transform(opt.image_size)
    dataset = _build_lmdb_dataset(opt.dataset_dir / 'val', log, transform=transform) 

    log.info(f"[Dataset] Built Custom dataset, size={len(dataset)}!")
    return dataset

class InpaintingVal10kSubset(Dataset):
    def __init__(self, opt, log, mask):
        super(InpaintingVal10kSubset, self).__init__()

        assert mask in ["center", "freeform1020", "freeform2030"]
        self.mask_type = mask
        self.dataset = build_lmdb_dataset_val10k(opt, log)

        from corruption.inpaint import get_center_mask, load_freeform_masks
        if self.mask_type == "center":
            self.mask = get_center_mask([opt.image_size, opt.image_size]) # [1,256,256]
        else:
            self.masks = load_freeform_masks(mask)[:,0,...] # [10000, 256, 256]
            assert len(self.dataset) == len(self.masks)

    def __len__(self):
        return self.dataset.__len__()

    def __getitem__(self, index):
        mask = self.mask if self.mask_type == "center" else self.masks[[index]]
        return *self.dataset[index], mask
