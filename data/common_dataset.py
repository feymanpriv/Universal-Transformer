"""Common Dataset."""

import sys
import os
import torch
import numpy as np
import random
from PIL import Image


class DataSet(torch.utils.data.Dataset):
    """Common dataset."""
    def __init__(self, data_path, split, transform = None):
        assert os.path.exists(data_path), "Data path '{}' not found".format(data_path)
        self._data_path, self._split = data_path, split
        self.transform = transform
        self._construct_imdb()
        
    def _construct_imdb(self):
        """Constructs the imdb."""
        # Compile the split data path
        self._imdb, self._class_ids = [], []
        with open(os.path.join(self._data_path, self._split), "r") as fin:
            for line in fin:
                info = line.strip().split(" ")
                im_dir, cont_id = info[0], info[1]
                im_path = os.path.join(self._data_path, im_dir)
                self._imdb.append({"im_path": im_path, "class": int(cont_id)})
                self._class_ids.append(int(cont_id))

    def __getitem__(self, index):
        # Load the image
        try:
            im = Image.open(self._imdb[index]["im_path"])
        except:
            #print('Error Loader: ', self._imdb[index]["im_path"])
            index = random.randint(0, 4221172)
            im = Image.open(self._imdb[index]["im_path"])
            #random_img = np.random.rand(384, 384, 3) * 255
            #im = Image.fromarray(np.uint8(random_img))
        im = im.convert('RGB')
        im = self.transform(im)
        
        label = self._imdb[index]["class"]
        return im, label

    def __len__(self):
        return len(self._imdb)