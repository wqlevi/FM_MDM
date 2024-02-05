import pickle
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import lmdb, six, os, io
from dataset.imagenet import build_train_transform, build_test_transform
from PIL import Image
class ImageFolderLMDB(Dataset):
    def __init__(self, path, transform, **kwargs):
        self.env = None
        self.db_path = path
        if 'train' in self.db_path:
            self.length = 1281167
        elif 'val' in self.db_path:
            self.length = 50000
        else:
            raise NotImplementedError
        self.transform = transform

    def _inti_lmdb(self):
        self.env = lmdb.open(self.db_path, subdir=os.path.isdir(self.db_path),
                             readonly=True, lock=False,
                             readahead=False, meminit=False)
        with self.env.begin(write=False) as txn:
            self.length = txn.stat()['entries']
            #self.keys = [key for key,_ in txn.cursor()] # [FIXME] hangs here
            self.keys = list(txn.cursor().iternext(values=False)) # [FIXME] hangs here

    def __getitem__(self, index):
        iimg, target = None, None
        if self.env is None:
            self._inti_lmdb()
        env = self.env
        with env.begin(write=False) as txn:
            byteflow = txn.get(self.keys[index])
        #unpacked = pickle.loads(byteflow)

        # load image
        #imgbuf = unpacked[0]
        #buf = six.BytesIO()
        #buf.write(imgbuf)
        #buf.seek(0)
        img = Image.open(io.BytesIO(byteflow)).convert('RGB')

        # load label
        #target = unpacked[1]

        if self.transform is not None:
            img = self.transform(img)
        return img#, target

    def __len__(self):
        return self.length

    def __repr__(self):
        return self.__class__.__name__ + ' (' + self.db_path + ')'

class Dset(Dataset):
    def __init__(self, root:os.PathLike, image_size = 256):
        self.files = sorted(root.glob("*.jpg"))
        self.transform = build_train_transform(image_size)
    def __len__(self):
        return len(self.files)
    def __getitem__(self, index):
        img = Image.open(self.files[index % len(self.files)])
        img_tensor = self.transform(img)
        return img_tensor, torch.tensor([0])
        

