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

                    # 3. Color and lighting changes (Day/Night/Shadows)
                  transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),

                    # 4. blury
                  transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.3),

                  transforms.ToTensor(),
                  normalize
                ])
          else:
                # only resize and normalize
              self.transform = transforms.Compose([
                  transforms.Resize((128, 128)),
                  transforms.ToTensor(),
                  normalize
                ])

        # Load labels
        with open(json_file, "r") as f:
            self.meta = json.load(f)
        self.keys = list(self.meta.keys())

        # Detect if input is a ZIP file
        self.is_zip = os.path.isfile(path) and (path.endswith('.zip') or zipfile.is_zipfile(path))

        # Zip handling variables
        self.zip_handle = None
        self.prefix = ""

        if self.is_zip:
            # Figuring out if images are in a subfolder inside the zip
            # e.g., mchar_train.zip -> mchar_train/000000.png
            with zipfile.ZipFile(path, 'r') as zf:
                names = zf.namelist()
                sample_key = self.keys[0] # e.g. "000000.png"

                if sample_key in names:
                    self.prefix = ""
                else:
                    # Check common prefix like "mchar_train/"
                    folder_name = os.path.basename(path).replace('.zip', '')
                    if f"{folder_name}/{sample_key}" in names:
                        self.prefix = f"{folder_name}/"
                    else:
                        # Fallback search
                        for n in names[:100]:
                            if n.endswith(sample_key):
                                self.prefix = n[:-len(sample_key)]
                                break

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        name = self.keys[idx] # e.g., "000000.png"

        if self.is_zip:
            # Open zip in the worker process (thread-safe way)
            if self.zip_handle is None:
                self.zip_handle = zipfile.ZipFile(self.path, 'r')

            try:
                full_path = self.prefix + name
                img_bytes = self.zip_handle.read(full_path)
                img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            except Exception:
                # Return black image if file is corrupt/missing
                img = Image.new('RGB', (128, 64))
        else:
            # Standard folder reading
            img_path = os.path.join(self.path, name)
            try:
                img = Image.open(img_path).convert("RGB")
            except:
                img = Image.new('RGB', (128, 64))

