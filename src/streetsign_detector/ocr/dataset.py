import json
import os
import zipfile
import io
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import torch

from streetsign_detector.constants import MAX_LEN, PADDING_TOKEN

class MCharDataset(Dataset):
    def __init__(self, path, json_file, transform=None, mode='train'): # new bool var mode for distinguish purposes on augmentation

        self.path = path
        if transform is not None:

          self.transform = transform
        else:
          normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # normalise the image

          if mode == 'train':
              self.transform = transforms.Compose([
                  transforms.Resize((128, 128)), # resize to fixed size

                    # 1. Geometric random rotate (-10 to 10 degrees)
                  transforms.RandomRotation(10),

                    # 2. Geometric random shifting
                  transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.9, 1.1)),


                  transforms.ToTensor(),
                  normalize
                ])


    def __len__(self):
        return len(self.keys)

